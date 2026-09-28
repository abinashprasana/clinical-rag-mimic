"""Retrieval variants behind one interface, all over the same chunks.

Every retriever takes a question and returns a ranked list of chunk indices
(at most k), with the same near duplicate filter the live pipeline applies.

* dense              core.retrieval.retrieve_chunks as the app runs it: the index on
                     disk (contextual text when CONTEXTUAL_INDEX is on) and the
                     header boost with stemmed matching
* dense_legacy       the behaviour before the evaluation work: plain chunk text
                     in the index and unstemmed header matching
* dense_no_boost     the same code path with the header boost off
* bm25               Okapi BM25 over raw lowercased chunk text, negation terms kept
* hybrid_rrf         dense and BM25 fused by reciprocal rank fusion (k = 60)
* hybrid_rrf_rerank  top 20 of the fused list reranked by a local cross encoder

Only `dense` is the application default. The others exist to be measured
side by side; nothing here changes what the app does.
"""
import re

from core.retrieval import _is_near_duplicate, _word_set, retrieve_chunks

VARIANTS = ('dense', 'dense_legacy', 'dense_no_boost', 'bm25', 'hybrid_rrf', 'hybrid_rrf_rerank')
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
    def __init__(self, embed_model, index, chunks, provenance, header_boost=True,
                 stem_headers=None, note_first=False):
        self.embed_model, self.index = embed_model, index
        self.chunks, self.provenance = chunks, provenance
        self.header_boost = header_boost
        self.options = {'stem_headers': stem_headers, 'note_first': note_first}

    supports_note_scope = True

    def retrieve(self, question, k=5, note=None):
        results = retrieve_chunks(
            question, self.embed_model, self.index, self.chunks, self.provenance,
            k=k, header_boost=self.header_boost, note_filter=note, **self.options,
        )
        return [r['chunk_idx'] for r in results]


class BM25Retriever:
    def __init__(self, chunks, index_texts=None):
        """index_texts, when given, is what BM25 scores (for example the
        contextual texts); chunks stay the texts used for de-duplication."""
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self.bm25 = BM25Okapi([bm25_tokens(c) for c in (index_texts or chunks)])

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


def _index_over(embed_model, texts):
    import faiss
    vectors = embed_model.encode(texts, show_progress_bar=False).astype('float32')
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


PRE_CONTEXTUAL_INDEX = 'faiss_index.pre_contextual.index'


def plain_index(embed_model, corpus):
    """FAISS index over the plain chunk text, as the index was built before
    contextual index text. Reuses the saved pre-change index when the corpus
    folder has one (the real index takes about 45 minutes to re-embed on a
    CPU), provided it covers the same number of chunks."""
    import os

    import faiss
    saved = os.path.join(corpus.output_dir or '', PRE_CONTEXTUAL_INDEX)
    if corpus.output_dir and os.path.exists(saved):
        index = faiss.read_index(saved)
        if index.ntotal == len(corpus.chunks):
            return index
    return _index_over(embed_model, corpus.chunks)


def contextual_index(embed_model, corpus):
    """FAISS index over the contextual texts (core.chunking.contextual_texts),
    built in memory with the same embedding model as the live index."""
    from core.chunking import contextual_texts
    return _index_over(embed_model, contextual_texts(corpus.chunks, corpus.provenance))


def build_retrievers(corpus, variants=VARIANTS, embed_model=None, index=None, cross_encoder=None):
    """Builds the requested variants over one corpus. Heavy models load only
    when a variant needs them."""
    needs_dense = any(v in variants for v in
                      ('dense', 'dense_legacy', 'dense_no_boost', 'hybrid_rrf', 'hybrid_rrf_rerank'))
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
        elif variant == 'dense_legacy':
            built[variant] = DenseRetriever(embed_model, plain_index(embed_model, corpus),
                                            corpus.chunks, corpus.provenance, stem_headers=False)
        else:
            raise ValueError(f'unknown retrieval variant {variant!r}')
    return built
