"""Contextual index text, stemmed header matching and note-first ranking.
Fabricated chunks only; a bag of words embedder keeps it model free."""
import faiss
import numpy as np

import config
from core.chunking import contextual_texts
from core.retrieval import _header_boost, _normalized_words, retrieve_chunks

CHUNKS = [
    '[History of Present Illness] A 70 year old presented with pneumonia. Cough for days.',
    '[Discharge Diagnosis] Primary diagnosis at discharge: community acquired pneumonia.',
    '[Discharge Medications] 1. Levofloxacin 750 mg PO daily 2. Metformin 1000 mg PO BID',
    '[History of Present Illness] A 60 year old presented with a hip fracture after a fall.',
    '[Discharge Medications] 1. Oxycodone 5 mg PO every 6 hours 2. Enoxaparin 40 mg daily',
]
PROVENANCE = [{'subject_id': 1, 'hadm_id': 10}] * 3 + [{'subject_id': 2, 'hadm_id': 20}] * 2
VOCAB = sorted({w for c in CHUNKS + ['medications discharged pneumonia patient']
                for w in _normalized_words(c)})


class BagOfWords:
    def encode(self, texts, **_):
        out = np.zeros((len(texts), len(VOCAB)), dtype='float32')
        for row, text in enumerate(texts):
            for w in _normalized_words(text):
                if w in VOCAB:
                    out[row, VOCAB.index(w)] += 1
        return out


def index_over(texts):
    vectors = BagOfWords().encode(texts)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def test_contextual_text_adds_note_context_and_keeps_chunks():
    texts = contextual_texts(CHUNKS, PROVENANCE)
    assert texts[2].startswith('(Note context: A 70 year old presented with pneumonia.')
    assert 'community acquired pneumonia' in texts[2] and texts[2].endswith(CHUNKS[2])
    assert 'hip fracture' in texts[4] and 'pneumonia' not in texts[4]
    assert contextual_texts(['no header chunk'], [{'hadm_id': 1}]) == ['no header chunk']


def test_stemmed_header_matching():
    question = _normalized_words('What was the patient discharged on?', stem=True)
    assert _header_boost(CHUNKS[1], question, stem=True) > 0
    unstemmed = _normalized_words('What was the patient discharged on?')
    assert _header_boost(CHUNKS[1], unstemmed) == 0


def test_contextual_index_finds_the_medication_chunk_of_the_right_patient():
    question = 'What medications was the pneumonia patient discharged on?'
    plain = retrieve_chunks(question, BagOfWords(), index_over(CHUNKS), CHUNKS, PROVENANCE, k=5,
                            stem_headers=True)
    contextual = retrieve_chunks(question, BagOfWords(), index_over(contextual_texts(CHUNKS, PROVENANCE)),
                                 CHUNKS, PROVENANCE, k=5, stem_headers=True)
    rank = lambda results: [r['chunk_idx'] for r in results].index(2)
    assert rank(contextual) <= rank(plain)
    assert all(r['chunk_text'] in CHUNKS for r in contextual)   # stored text stays plain


def test_note_first_keeps_results_inside_the_top_note_first():
    question = 'What medications was the pneumonia patient discharged on?'
    results = retrieve_chunks(question, BagOfWords(), index_over(CHUNKS), CHUNKS, PROVENANCE, k=3,
                              note_first=True)
    assert {r['hadm_id'] for r in results} == {10}


def test_note_filter_answers_from_one_admission_only():
    question = 'What medications was the patient discharged on?'
    index = index_over(CHUNKS)
    scoped = retrieve_chunks(question, BagOfWords(), index, CHUNKS, PROVENANCE, k=5, note_filter='20')
    assert scoped and {r['hadm_id'] for r in scoped} == {20}
    assert scoped[0]['chunk_idx'] == 4   # that admission's medication list
    unscoped = retrieve_chunks(question, BagOfWords(), index, CHUNKS, PROVENANCE, k=5)
    assert {r['hadm_id'] for r in unscoped} == {10, 20}
    assert retrieve_chunks(question, BagOfWords(), index, CHUNKS, PROVENANCE, k=5, note_filter='999') == []


def test_defaults_follow_config():
    assert config.CONTEXTUAL_INDEX is True and config.RETRIEVAL_STEM_HEADERS is True
    assert config.GENERATION_TOP_K >= 1
