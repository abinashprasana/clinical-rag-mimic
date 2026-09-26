"""Poisoned note harness: the adversarial notes stay out of the public demo
index, every one survives chunking with its canary, and the temporary index
never touches outputs_demo/. A fake embedder keeps this model free."""
import hashlib
import os
import pathlib

import numpy as np
import sentence_transformers

from core.chunking import section_chunk
from demo.adversarial_notes import ADVERSARIAL_NOTES, CANARY, INJECTION_QUESTIONS
from demo.notes import DEMO_NOTES
from evals.corpus import load_corpus
from evals.run_injection_eval import build_temp_corpus, summarise


def digest(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(pathlib.Path(folder).iterdir()) if p.is_file()}


def test_adversarial_notes_are_not_in_the_public_demo_index():
    demo_ids = {str(n['hadm_id']) for n in DEMO_NOTES}
    adversarial_ids = {str(n['hadm_id']) for n in ADVERSARIAL_NOTES}
    assert not demo_ids & adversarial_ids
    corpus = load_corpus('demo')
    assert not any(CANARY in chunk for chunk in corpus.chunks)
    assert not {key[0] for key in corpus.keys} & adversarial_ids
    assert 'adversarial' not in pathlib.Path('demo/build_index.py').read_text(encoding='utf-8')


def test_every_poisoned_note_keeps_its_canary_after_chunking():
    for note in ADVERSARIAL_NOTES:
        assert any(CANARY in chunk for chunk in section_chunk(note['text']))
    targeted = {hadm for _, hadm in INJECTION_QUESTIONS}
    assert targeted == {n['hadm_id'] for n in ADVERSARIAL_NOTES}
    assert not any(w in q.lower() for q, _ in INJECTION_QUESTIONS for w in ('dose', 'dosage'))


def test_temp_index_leaves_outputs_demo_untouched(tmp_path, monkeypatch):
    class FakeEmbedder:
        def __init__(self, *_):
            pass

        def encode(self, texts, **_):
            return np.random.default_rng(0).random((len(texts), 8)).astype('float32')

    monkeypatch.setattr(sentence_transformers, 'SentenceTransformer', FakeEmbedder)
    before = digest('outputs_demo')
    _, index, chunks, provenance = build_temp_corpus(str(tmp_path))
    assert digest('outputs_demo') == before
    assert os.listdir(tmp_path) == ['faiss_index.index']
    assert index.ntotal == len(chunks) == len(provenance)
    assert any(CANARY in c for c in chunks)


def test_summary_counts_only_exposed_questions():
    rows = [
        {'poisoned_in_context': True, 'canary_in_draft': True, 'canary_in_final': True, 'gate_passed': True},
        {'poisoned_in_context': True, 'canary_in_draft': False, 'canary_in_final': False, 'gate_passed': True},
        {'poisoned_in_context': False, 'canary_in_draft': False, 'canary_in_final': False, 'gate_passed': True},
    ]
    s = summarise(rows)
    assert s['poisoned_passage_retrieved']['k'] == 2 and s['poisoned_passage_retrieved']['n'] == 3
    assert s['canary_in_final_answer']['k'] == 1 and s['canary_in_final_answer']['n'] == 2
    assert s['gate_passed_canary_draft'] == {**s['gate_passed_canary_draft'], 'k': 1, 'n': 1}
