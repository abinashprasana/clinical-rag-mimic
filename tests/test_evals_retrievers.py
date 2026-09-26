"""Retriever variants on a tiny fabricated corpus with a fake embedder, so no
model downloads are needed."""
import faiss
import numpy as np

from evals.corpus import Corpus
from evals.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRRFRetriever,
    RerankRetriever,
    bm25_tokens,
    dedup,
    rrf_fuse,
)

CHUNKS = [
    '[Discharge Medications] furosemide 40 mg daily and metoprolol 25 mg twice daily',
    '[History of Present Illness] patient denies chest pain and reports leg swelling',
    '[Physical Exam] lungs clear, no edema, heart regular rhythm on exam today',
    '[Discharge Diagnosis] congestive heart failure exacerbation with volume overload',
]
PROVENANCE = [{'subject_id': 1, 'hadm_id': 10}] * 4
CORPUS = Corpus('tiny', CHUNKS, PROVENANCE, output_dir=None)
VOCAB = sorted({t for c in CHUNKS for t in bm25_tokens(c)} | {'medications', 'pain'})


class BagOfWordsModel:
    """Fake sentence embedder: bag of words over a fixed vocabulary."""
    def encode(self, texts, **_):
        out = np.zeros((len(texts), len(VOCAB)), dtype='float32')
        for row, text in enumerate(texts):
            for token in bm25_tokens(text):
                if token in VOCAB:
                    out[row, VOCAB.index(token)] += 1
        return out


def dense_index():
    vectors = BagOfWordsModel().encode(CHUNKS)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def test_bm25_keeps_negation_terms():
    assert 'no' in bm25_tokens('No edema.') and 'denies' in bm25_tokens('Denies pain')
    ranking = BM25Retriever(CHUNKS).retrieve('denies chest pain', k=2)
    assert ranking[0] == 1


def test_rrf_fuse_matches_hand_computation():
    # doc 7: 1/61 + 1/62; doc 8: 1/62 + 1/61; doc 9: 1/63 only
    fused = rrf_fuse([[7, 8, 9], [8, 7]])
    assert set(fused[:2]) == {7, 8} and fused[2] == 9
    weighted = rrf_fuse([[1], [2]], weights=[1.0, 2.0])
    assert weighted == [2, 1]


def test_dedup_drops_near_duplicate_windows():
    chunks = ['a b c d e f g h', 'a b c d e f g x', 'totally different words here']
    assert dedup([0, 1, 2], chunks, k=5) == [0, 2]


def test_header_boost_switch_changes_only_the_ordering():
    model, index = BagOfWordsModel(), dense_index()
    question = 'discharge medications exam'
    boosted = DenseRetriever(model, index, CHUNKS, PROVENANCE).retrieve(question, k=4)
    raw = DenseRetriever(model, index, CHUNKS, PROVENANCE, header_boost=False).retrieve(question, k=4)
    assert sorted(boosted) == sorted(raw)
    assert boosted[0] == 0


def test_hybrid_and_rerank_return_k_distinct_indices():
    model, index = BagOfWordsModel(), dense_index()
    dense = DenseRetriever(model, index, CHUNKS, PROVENANCE)
    hybrid = HybridRRFRetriever(dense, BM25Retriever(CHUNKS))
    result = hybrid.retrieve('heart failure furosemide', k=3)
    assert len(result) == len(set(result)) == 3

    class ReverseLengthReranker:
        def predict(self, pairs):
            return [-len(text) for _, text in pairs]

    reranked = RerankRetriever(hybrid, ReverseLengthReranker()).retrieve('heart failure', k=2)
    shortest = sorted(range(4), key=lambda i: len(CHUNKS[i]))[:2]
    assert reranked == shortest
    assert [CORPUS.keys[i][1] for i in reranked]
