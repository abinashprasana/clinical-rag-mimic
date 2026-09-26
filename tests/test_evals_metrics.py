"""Metric functions on tiny hand-made rankings. Values are worked out by hand."""
import math

import pytest

from evals.metrics import (
    bootstrap_ci,
    cohen_kappa,
    confusion_matrix,
    hit_at_k,
    ndcg_at_k,
    note_hit_at_k,
    paired_bootstrap_diff,
    percentile,
    query_metrics,
    recall_at_k,
    reciprocal_rank,
    wilson_ci,
)

A, B, C = ('1', 'Discharge Medications'), ('1', 'Allergies'), ('2', 'Discharge Medications')
X, Y = ('3', 'Physical Exam'), ('1', 'Physical Exam')


def test_hit_and_recall_at_k():
    ranking = [X, A, X, B, X]
    gold = {A, B}
    assert hit_at_k(ranking, gold, 1) == 0.0
    assert hit_at_k(ranking, gold, 2) == 1.0
    assert recall_at_k(ranking, gold, 1) == 0.0
    assert recall_at_k(ranking, gold, 3) == 0.5
    assert recall_at_k(ranking, gold, 5) == 1.0


def test_duplicate_chunks_of_one_entry_count_once():
    # two sub-chunks of the same section must not double count
    assert recall_at_k([A, A, X], {A, B}, 3) == 0.5
    assert ndcg_at_k([A, A], {A, B}, 5) == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_reciprocal_rank():
    assert reciprocal_rank([X, X, A], {A}) == pytest.approx(1 / 3)
    assert reciprocal_rank([A], {A}) == 1.0
    assert reciprocal_rank([X, Y], {A}) == 0.0


def test_ndcg_hand_values():
    assert ndcg_at_k([A, X, X, X, X], {A}, 5) == 1.0
    # single gold at rank 2: (1/log2(3)) / 1
    assert ndcg_at_k([X, A], {A}, 5) == pytest.approx(1 / math.log2(3))
    # two gold at ranks 1 and 3; ideal is ranks 1 and 2
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k([A, X, B], {A, B}, 5) == pytest.approx(expected)
    assert ndcg_at_k([X, X, X, X, X, A], {A}, 5) == 0.0


def test_note_hit_separates_wrong_section_from_wrong_patient():
    assert note_hit_at_k([Y], {A}, 5) == 1.0   # right note, wrong section
    assert note_hit_at_k([C], {A}, 5) == 0.0   # wrong note


def test_empty_gold_is_rejected():
    with pytest.raises(ValueError):
        recall_at_k([A], set(), 5)


def test_query_metrics_keys():
    m = query_metrics([X, A], {A})
    assert m == {
        'hit@5': 1.0, 'recall@1': 0.0, 'recall@3': 1.0, 'recall@5': 1.0,
        'mrr': 0.5, 'ndcg@5': pytest.approx(1 / math.log2(3)), 'note_hit@5': 1.0,
    }


def test_bootstrap_is_seeded_and_brackets_the_mean():
    values = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
    first, second = bootstrap_ci(values), bootstrap_ci(values)
    assert first == second
    mean, low, high = first
    assert mean == 0.7 and low <= mean <= high and 0 <= low and high <= 1
    assert bootstrap_ci([1, 1, 1]) == (1.0, 1.0, 1.0)


def test_paired_bootstrap_uses_differences():
    a = [1, 1, 1, 0, 1]
    assert paired_bootstrap_diff(a, a) == (0.0, 0.0, 0.0)
    mean, low, high = paired_bootstrap_diff([1] * 10, [0] * 10)
    assert (mean, low, high) == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1], [1, 0])


def test_wilson_interval():
    rate, low, high = wilson_ci(3, 3)
    assert rate == 1.0 and high == 1.0 and low == pytest.approx(0.4385, abs=1e-3)
    rate, low, high = wilson_ci(0, 8)
    assert rate == 0.0 and low == 0.0 and high == pytest.approx(0.3244, abs=1e-3)
    assert math.isnan(wilson_ci(0, 0)[0])


def test_cohen_kappa_textbook_example():
    # 50 items: both yes 20, a yes/b no 5, a no/b yes 10, both no 15.
    # po = 0.7, pe = 0.5*0.6 + 0.5*0.4 = 0.5, kappa = 0.4
    a = ['y'] * 25 + ['n'] * 25
    b = ['y'] * 20 + ['n'] * 5 + ['y'] * 10 + ['n'] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert cohen_kappa(['y', 'n'], ['y', 'n']) == 1.0
    assert math.isnan(cohen_kappa(['y', 'y'], ['y', 'y']))


def test_confusion_matrix_and_percentile():
    m = confusion_matrix(['s', 's', 'u'], ['s', 'u', 'u'], ['s', 'u'])
    assert m == {'s': {'s': 1, 'u': 1}, 'u': {'s': 0, 'u': 1}}
    assert percentile([1, 2, 3, 4], 50) == 2.5
