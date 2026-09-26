"""The CI gate fails on a degraded retriever and on the guard conditions.
Uses the fabricated demo corpus with an oracle and a shuffling retriever, so
no models load. Marking questions reviewed here is an in-memory fixture."""
import json
import random

import pytest

import evals.check_regression as gate
from evals.corpus import load_corpus
from evals.golden import GOLDEN_FILES, load_questions

CORPUS = load_corpus('demo')
QUESTIONS = [q for q in load_questions(GOLDEN_FILES['demo']) if q['expected_behavior'] == 'answer']
THRESHOLDS = {'recall@5': 0.02, 'mrr': 0.03}


class OracleRetriever:
    def retrieve(self, question, k=5):
        q = next(q for q in QUESTIONS if q['question'] == question)
        gold = [i for r in q['relevant'] for i in CORPUS.chunk_indices(r['note_id'], r['section'])]
        return gold[:k]


class ShuffledRetriever:
    """Degraded retriever: a seeded random ranking of the whole corpus."""
    def __init__(self, seed=3):
        self.rng = random.Random(seed)

    def retrieve(self, question, k=5):
        order = list(range(len(CORPUS.chunks)))
        self.rng.shuffle(order)
        return order[:k]


def test_gate_passes_against_itself_and_fails_on_a_shuffled_ranking():
    baseline = gate.measure(OracleRetriever(), CORPUS, QUESTIONS)
    assert gate.compare(baseline, baseline, THRESHOLDS)[0] is True
    degraded = gate.measure(ShuffledRetriever(), CORPUS, QUESTIONS)
    passed, messages = gate.compare(degraded, baseline, THRESHOLDS)
    assert passed is False
    assert any(m.startswith('FAIL  recall@5') for m in messages)


def test_gate_tolerates_one_lost_question_and_fails_at_two():
    baseline = gate.measure(OracleRetriever(), CORPUS, QUESTIONS)
    assert len(QUESTIONS) == 70
    one_lost = {**baseline, 'recall@5': baseline['recall@5'] - 1 / len(QUESTIONS)}
    two_lost = {**baseline, 'recall@5': baseline['recall@5'] - 2 / len(QUESTIONS)}
    assert gate.compare(one_lost, baseline, THRESHOLDS)[0] is True
    assert gate.compare(two_lost, baseline, THRESHOLDS)[0] is False


def test_gate_fails_when_question_set_changes():
    baseline = gate.measure(OracleRetriever(), CORPUS, QUESTIONS)
    current = {**baseline, 'question_ids': baseline['question_ids'][:-1]}
    passed, messages = gate.compare(current, baseline, THRESHOLDS)
    assert passed is False and 'update-baseline' in messages[0]


def test_thresholds_file_matches_the_documented_defaults():
    with open(gate.THRESHOLDS_PATH, encoding='utf-8') as f:
        thresholds = json.load(f)
    assert thresholds['recall@5'] == 0.02 and thresholds['mrr'] == 0.03


@pytest.mark.parametrize('env', ['CI', 'GITHUB_ACTIONS'])
def test_update_baseline_refuses_in_ci(monkeypatch, env):
    monkeypatch.setenv(env, 'true')
    assert gate.main(['--update-baseline']) == 1


def test_gate_fails_loudly_without_reviewed_questions(monkeypatch):
    monkeypatch.delenv('CI', raising=False)
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    monkeypatch.setattr(gate, 'reviewed_retrieval_questions', lambda corpus: [])
    assert gate.main([]) == 1
