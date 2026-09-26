"""CI regression gate: default retriever on the reviewed demo questions,
compared with the committed baseline.

    python -m evals.check_regression                    # exits 1 on a regression
    python -m evals.check_regression --update-baseline  # deliberate, local only

The gate fails when recall@5 or MRR drops by more than the limits in
evals/thresholds.json, when no reviewed questions exist, when no baseline
exists, or when the reviewed question set differs from the one the baseline
was built on (update the baseline on purpose after reviewing questions).
"""
import argparse
import json
import os
import sys

from evals import common
from evals.corpus import load_corpus
from evals.golden import GOLDEN_FILES, load_questions, select, validate
from evals.retrievers import DEFAULT_VARIANT, build_retrievers
from evals.run_retrieval_eval import evaluate_variant

BASELINE_PATH = os.path.join('evals', 'baseline_demo.json')
THRESHOLDS_PATH = os.path.join('evals', 'thresholds.json')
GATED = ('recall@5', 'mrr')


def measure(retriever, corpus, questions):
    records = evaluate_variant(retriever, corpus, questions)
    return {
        'retriever': DEFAULT_VARIANT,
        'question_ids': sorted(q['id'] for q in questions),
        'n': len(records),
        **{m: sum(r[m] for r in records) / len(records) for m in GATED},
    }


def compare(current, baseline, thresholds):
    """(passed, messages)."""
    if current['question_ids'] != baseline['question_ids']:
        return False, [('The reviewed question set differs from the baseline. Review the change, then run '
                        'python -m evals.check_regression --update-baseline and commit the new baseline.')]
    passed, messages = True, []
    for metric in GATED:
        drop = baseline[metric] - current[metric]
        limit = thresholds[metric]
        status = 'FAIL' if drop > limit + 1e-12 else 'ok'
        passed &= status == 'ok'
        messages.append(f'{status}  {metric}: baseline {baseline[metric]:.4f}, now {current[metric]:.4f}, '
                        f'drop {drop:+.4f} (limit {limit})')
    return passed, messages


def reviewed_retrieval_questions(corpus):
    questions = load_questions(GOLDEN_FILES['demo'])
    errors, _ = validate(questions, corpus)
    if errors:
        raise SystemExit(f'{len(errors)} golden set validation errors; run python -m evals.golden')
    return [q for q in select(questions) if q['expected_behavior'] == 'answer']


def in_ci():
    return any(os.environ.get(name, '').lower() in ('1', 'true') for name in ('CI', 'GITHUB_ACTIONS'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--update-baseline', action='store_true')
    args = parser.parse_args(argv)

    if args.update_baseline and in_ci():
        print('Refusing to update the baseline in CI. Update it locally and commit it on purpose.')
        return 1
    corpus = load_corpus('demo')
    questions = reviewed_retrieval_questions(corpus)
    if not questions:
        print('No reviewed retrieval questions in evals/golden/demo_questions.jsonl. '
              'The gate needs reviewed questions and a baseline; it does not fall back to drafts.')
        return 1
    current = measure(build_retrievers(corpus, [DEFAULT_VARIANT])[DEFAULT_VARIANT], corpus, questions)

    baseline = None
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH, encoding='utf-8') as f:
            baseline = json.load(f)

    if args.update_baseline:
        new = {**current, 'meta': common.run_metadata('demo', False, current['n'])}
        print('Baseline diff:')
        for metric in GATED:
            old = f'{baseline[metric]:.4f}' if baseline else 'none'
            print(f'  {metric}: {old} -> {new[metric]:.4f}')
        if baseline and baseline['question_ids'] != new['question_ids']:
            added = sorted(set(new['question_ids']) - set(baseline['question_ids']))
            removed = sorted(set(baseline['question_ids']) - set(new['question_ids']))
            print(f'  questions added: {added or "none"}; removed: {removed or "none"}')
        common.write_json(BASELINE_PATH, new)
        print(f'Wrote {BASELINE_PATH}. Commit it deliberately.')
        return 0

    if baseline is None:
        print(f'No baseline at {BASELINE_PATH}. Create one locally with --update-baseline and commit it.')
        return 1
    with open(THRESHOLDS_PATH, encoding='utf-8') as f:
        thresholds = json.load(f)
    passed, messages = compare(current, baseline, thresholds)
    print(f'Retrieval regression gate: {DEFAULT_VARIANT} retriever, {current["n"]} reviewed questions.')
    print('\n'.join(messages))
    print('PASSED' if passed else 'FAILED')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
