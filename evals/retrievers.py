"""Retrieval variants behind one interface, all over the same chunks.

Every retriever takes a question and returns a ranked list of chunk indices
(at most k), with the same near duplicate filter the live pipeline applies.

* dense              core.retrieval.retrieve_chunks as used today (header boost on)
* dense_no_boost     the same code path with the header boost off
* bm25               Okapi BM25 over raw lowercased chunk text, negation terms kept
* hybrid_rrf         dense and BM25 fused by reciprocal rank fusion (k = 60)
* hybrid_rrf_rerank  top 20 of the fused list reranked by a local cross encoder

Only `dense` is the application default. The others exist to be measured
side by side; nothing here changes what the app does.
"""
import re

from core.retrieval import _is_near_duplicate, _word_set, retrieve_chunks

VARIANTS = ('dense', 'dense_no_boost', 'bm25', 'hybrid_rrf', 'hybrid_rrf_rerank')
DEFAULT_VARIANT = 'dense'
RRF_K = 60
FUSION_DEPTH = 50
RERANK_POOL = 20
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

_TOKEN_RE = re.compile(r'[a-z0-9]+')


def bm25_tokens(text):
    """Lowercase alphanumeric tokens with no stopword removal, so "no", "not"
    and "denies" stay searchable."""
    return _TOKEN_RE.findall(text.lower())


def dedup(indices, chunks, k):
    kept, kept_words = [], []
    for i in indices:
        words = _word_set(chunks[i])
        if _is_near_duplicate(words, kept_words):
            continue
        kept_words.append(words)
        kept.append(i)
        if len(kept) == k:
            break
    return kept


def rrf_fuse(rankings, weights=None, k=RRF_K):
    """Weighted reciprocal rank fusion (Cormack, Clarke and Buettcher 2009).
    `rankings` is a list of ranked index lists; ties keep first-seen order."""
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError('one weight per ranking')
    scores = {}
    for ranking, weight in zip(rankings, weights):
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + weight / (k + rank)
    return sorted(scores, key=lambda idx: -scores[idx])


class DenseRetriever:
    def __init__(self, embed_model, index, chunks, provenance, header_boost=True):
        self.embed_model, self.index = embed_model, index
        self.chunks, self.provenance = chunks, provenance
        self.header_boost = header_boost

    def retrieve(self, question, k=5):
        results = retrieve_chunks(
            question, self.embed_model, self.index, self.chunks, self.provenance,
            k=k, header_boost=self.header_boost,
        )
        return [r['chunk_idx'] for r in results]


class BM25Retriever:
    def __init__(self, chunks):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self.bm25 = BM25Okapi([bm25_tokens(c) for c in chunks])

    def ranking(self, question, depth):
        scores = self.bm25.get_scores(bm25_tokens(question))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return order[:depth]

    def retrieve(self, question, k=5):
        return dedup(self.ranking(question, max(FUSION_DEPTH, k * 3)), self.chunks, k)


class HybridRRFRetriever:
    def __init__(self, dense, bm25, weights=(1.0, 1.0), depth=FUSION_DEPTH):
        self.dense, self.bm25 = dense, bm25
        self.weights, self.depth = list(weights), depth
        self.chunks = bm25.chunks

    def fused(self, question):
        return rrf_fuse(
            [self.dense.retrieve(question, k=self.depth), self.bm25.ranking(question, self.depth)],
            self.weights,
        )

    def retrieve(self, question, k=5):
        return dedup(self.fused(question), self.chunks, k)


class RerankRetriever:
    def __init__(self, hybrid, cross_encoder, pool=RERANK_POOL):
        self.hybrid, self.cross_encoder, self.pool = hybrid, cross_encoder, pool

    def retrieve(self, question, k=5):
        chunks = self.hybrid.chunks
        pool = dedup(self.hybrid.fused(question), chunks, self.pool)
        scores = self.cross_encoder.predict([(question, chunks[i]) for i in pool])
        order = sorted(range(len(pool)), key=lambda j: -float(scores[j]))
        return [pool[j] for j in order[:k]]


def build_retrievers(corpus, variants=VARIANTS, embed_model=None, index=None, cross_encoder=None):
    """Builds the requested variants over one corpus. Heavy models load only
    when a variant needs them."""
    needs_dense = any(v in variants for v in ('dense', 'dense_no_boost', 'hybrid_rrf', 'hybrid_rrf_rerank'))
    if needs_dense and (embed_model is None or index is None):
        import faiss
        from sentence_transformers import SentenceTransformer

        import config
        embed_model = embed_model or SentenceTransformer(config.EMBEDDING_MODEL)
        index = index or faiss.read_index(f'{corpus.output_dir}/faiss_index.index')

    built = {}
    dense = DenseRetriever(embed_model, index, corpus.chunks, corpus.provenance) if needs_dense else None
    bm25 = BM25Retriever(corpus.chunks) if any(v.startswith(('bm25', 'hybrid')) for v in variants) else None
    for variant in variants:
        if variant == 'dense':
            built[variant] = dense
        elif variant == 'dense_no_boost':
            built[variant] = DenseRetriever(
                embed_model, index, corpus.chunks, corpus.provenance, header_boost=False)
        elif variant == 'bm25':
            built[variant] = bm25
        elif variant == 'hybrid_rrf':
            built[variant] = HybridRRFRetriever(dense, bm25)
        elif variant == 'hybrid_rrf_rerank':
            if cross_encoder is None:
                from sentence_transformers import CrossEncoder
                cross_encoder = CrossEncoder(RERANKER_MODEL)
            built[variant] = RerankRetriever(HybridRRFRetriever(dense, bm25), cross_encoder)
        else:
            raise ValueError(f'unknown retrieval variant {variant!r}')
    return built
