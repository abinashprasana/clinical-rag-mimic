"""Ranking metrics, confidence intervals and agreement statistics.

Relevance is judged per gold entry, a (note_id, section) pair. A retrieved
chunk is relevant when its own (note_id, section) matches a gold entry, so
re-chunking never invalidates the gold set.

* hit@k     1 if any of the top k chunks is relevant.
* recall@k  share of gold entries covered by at least one of the top k chunks.
* MRR       1 / rank of the first relevant chunk in the retrieved list, 0 if none.
* nDCG@k    binary gains; a second chunk from an already credited gold entry
            gains nothing, and the ideal DCG assumes min(|gold|, k) hits.
* note hit@k  diagnostic: 1 if any top k chunk comes from a gold note, in any
            section. It separates "wrong patient" from "right patient, wrong section".

Intervals: percentile bootstrap over questions (Sakai 2006), a paired
bootstrap on per question differences between two systems (Smucker, Allan
and Carterette 2007), and the Wilson score interval for binary rates, which
keeps a sensible width at 0 of n or n of n (Brown, Cai and DasGupta 2001).
"""
import math

import numpy as np

N_RESAMPLES = 1000
SEED = 20260926
Z95 = 1.959963984540054


def _first_hits(retrieved, gold, k):
    """Positions (0 based) in the top k that credit a not yet credited gold entry."""
    credited, positions = set(), []
    for pos, key in enumerate(retrieved[:k]):
        if key in gold and key not in credited:
            credited.add(key)
            positions.append(pos)
    return positions


def hit_at_k(retrieved, gold, k):
    return float(any(key in gold for key in retrieved[:k]))


def recall_at_k(retrieved, gold, k):
    if not gold:
        raise ValueError('recall needs at least one gold entry')
    return len(_first_hits(retrieved, gold, k)) / len(gold)


def reciprocal_rank(retrieved, gold):
    for pos, key in enumerate(retrieved):
        if key in gold:
            return 1.0 / (pos + 1)
    return 0.0


def ndcg_at_k(retrieved, gold, k):
    if not gold:
        raise ValueError('nDCG needs at least one gold entry')
    dcg = sum(1.0 / math.log2(pos + 2) for pos in _first_hits(retrieved, gold, k))
    ideal = sum(1.0 / math.log2(pos + 2) for pos in range(min(len(gold), k)))
    return dcg / ideal


def note_hit_at_k(retrieved, gold, k):
    notes = {note for note, _ in gold}
    return float(any(note in notes for note, _ in retrieved[:k]))


def query_metrics(retrieved, gold):
    """All per question retrieval metrics. `retrieved` is a ranked list of
    (note_id, section) keys, `gold` a set of the same."""
    gold = set(gold)
    return {
        'hit@5': hit_at_k(retrieved, gold, 5),
        'recall@1': recall_at_k(retrieved, gold, 1),
        'recall@3': recall_at_k(retrieved, gold, 3),
        'recall@5': recall_at_k(retrieved, gold, 5),
        'mrr': reciprocal_rank(retrieved, gold),
        'ndcg@5': ndcg_at_k(retrieved, gold, 5),
        'note_hit@5': note_hit_at_k(retrieved, gold, 5),
    }


def bootstrap_ci(values, n_resamples=N_RESAMPLES, seed=SEED, alpha=0.05):
    """(mean, low, high) with a percentile bootstrap over the values."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float('nan'),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(low), float(high)


def paired_bootstrap_diff(a, b, n_resamples=N_RESAMPLES, seed=SEED, alpha=0.05):
    """(mean of a - b, low, high), resampling questions jointly so each
    question keeps its pair. An interval that excludes 0 is a difference."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError('paired bootstrap needs equal length inputs')
    return bootstrap_ci(a - b, n_resamples, seed, alpha)


def wilson_ci(successes, n, z=Z95):
    """(rate, low, high) for a binary rate."""
    if n == 0:
        return (float('nan'),) * 3
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q)) if len(values) else float('nan')


def cohen_kappa(labels_a, labels_b):
    """Cohen's kappa for two raters over the same items (Cohen 1960).
    Returns nan when agreement by chance is total (both raters constant
    and equal), since kappa is undefined there."""
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError('kappa needs two equal, non empty label lists')
    n = len(labels_a)
    categories = sorted(set(labels_a) | set(labels_b), key=str)
    observed = sum(x == y for x, y in zip(labels_a, labels_b)) / n
    expected = sum(
        (labels_a.count(c) / n) * (labels_b.count(c) / n) for c in categories
    )
    if expected == 1:
        return float('nan')
    return (observed - expected) / (1 - expected)


def confusion_matrix(truth, predicted, labels):
    """{truth_label: {predicted_label: count}}."""
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(truth, predicted):
        matrix[t][p] += 1
    return matrix
