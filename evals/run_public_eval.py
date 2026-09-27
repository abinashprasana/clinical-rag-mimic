"""Golden set evaluation of the public Vercel runtime (demo/runtime.py), which
retrieves with Gemini embeddings and answers with Gemini. Fabricated notes
only; this module reads demo/runtime.py and never changes it.

    python -m evals.run_public_eval
    python -m evals.run_public_eval --pause 6 --max-fallbacks 3

Scores the reviewed answer and unanswerable questions: recall@5 over the
passages the runtime cites, answer correctness and refusal accuracy. The
runtime quietly falls back to an offline keyword method when a Gemini call
fails (for example when the free quota runs out), so every answer records
which path produced it. Only Gemini answers are scored; the run stops after
--max-fallbacks fallbacks in a row, since that means the quota is spent.
"""
import argparse
import os
import sys
import time

from evals import common
from evals.corpus import load_corpus
from evals.golden import GOLDEN_FILES, contains, load_questions, select
from evals.metrics import bootstrap_ci, query_metrics, wilson_ci

GEMINI_MARKER = 'via Gemini embeddings'


def via_gemini(result):
    return GEMINI_MARKER in (result.get('tool_used') or '')


def score_answer(question, answer):
    return (all(contains(answer, t) for t in question['must_contain'])
            and not any(contains(answer, t) for t in question['must_not_contain']))


def is_refusal(answer):
    lowered = (answer or '').lower()
    return 'cannot find this information' in lowered or 'could not confirm' in lowered


def run(questions, corpus, runner, pause, max_fallbacks, sleep=time.sleep):
    records, fallbacks_in_row = [], 0
    for q in questions:
        result = runner(q['question'])
        gemini = via_gemini(result)
        answer = result.get('final_answer') or ''
        keys = [corpus.keys[c['chunk_idx']] for c in result.get('citations', [])]
        record = {'id': q['id'], 'category': q['category'], 'behavior': q['expected_behavior'],
                  'gemini': gemini}
        if q['expected_behavior'] == 'answer':
            gold = {(str(r['note_id']), r['section']) for r in q['relevant']}
            record['recall@5'] = query_metrics(keys, gold)['recall@5']
            record['correct'] = score_answer(q, answer) and not is_refusal(answer)
        else:
            record['correct'] = is_refusal(answer)
        records.append(record)
        fallbacks_in_row = 0 if gemini else fallbacks_in_row + 1
        if fallbacks_in_row >= max_fallbacks:
            break
        sleep(pause)
    return records


def summarise(records):
    scored = [r for r in records if r['gemini']]
    answer = [r for r in scored if r['behavior'] == 'answer']
    refuse = [r for r in scored if r['behavior'] == 'refuse']

    def rate(rows):
        k, n = sum(r['correct'] for r in rows), len(rows)
        p, low, high = wilson_ci(k, n)
        return {'k': k, 'n': n, 'rate': p, 'low': low, 'high': high} if n else None

    mean, low, high = bootstrap_ci([r['recall@5'] for r in answer])
    return {
        'attempted': len(records),
        'answered_by_gemini': len(scored),
        'fallback_answers_excluded': len(records) - len(scored),
        'recall@5': {'mean': mean, 'low': low, 'high': high} if answer else None,
        'answer_correctness': rate(answer),
        'refusal_accuracy': rate(refuse),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--pause', type=float, default=5.0, help='seconds between questions (free tier pacing)')
    parser.add_argument('--max-fallbacks', type=int, default=3)
    parser.add_argument('--split', choices=['all', 'dev', 'test'], default='all')
    parser.add_argument('--file', help='golden file (default: the 100 demo questions)')
    args = parser.parse_args(argv)

    import config
    if not config.GEMINI_API_KEY:
        print('GEMINI_API_KEY is not set, so the public runtime would only use its offline fallback. Nothing to score.')
        return 2
    from demo import runtime
    questions = [q for q in select(load_questions(args.file or GOLDEN_FILES['demo']), split=args.split)
                 if q['expected_behavior'] in ('answer', 'refuse')]
    records = run(questions, load_corpus('demo'), runtime.run_turn, args.pause, args.max_fallbacks)
    summary = summarise(records)
    result = {'meta': common.run_metadata('demo', False, len(questions)),
              'runtime': runtime.PUBLIC_GENERATOR_LABEL, **summary}
    tag = '' if not args.file else '_' + os.path.basename(args.file).split('.')[0]
    common.write_json(os.path.join(common.RESULTS_DIR, f'public_demo{tag}.json'), result)
    common.write_jsonl(os.path.join(common.DEMO_RUNS_DIR, 'public_records.jsonl'), records)

    def pct(row):
        return 'n/a' if not row else f"{row['k']}/{row['n']} = {row['rate']:.0%} [{row['low']:.0%}, {row['high']:.0%}]"
    r5 = summary['recall@5']
    print(f"Public runtime ({result['runtime']}), {len(questions)} questions, "
          f"{summary['answered_by_gemini']} answered by Gemini, {summary['fallback_answers_excluded']} fallback "
          'answers excluded.')
    print(common.text_table(['measure', 'result'], [
        ['recall@5 (cited passages)', 'n/a' if not r5 else common.fmt_ci(r5['mean'], r5['low'], r5['high'])],
        ['answer correctness', pct(summary['answer_correctness'])],
        ['refusal accuracy', pct(summary['refusal_accuracy'])],
    ]))
    if len(records) < len(questions):
        print(f'Stopped after {len(records)} of {len(questions)} questions: '
              f'{args.max_fallbacks} fallbacks in a row, so the Gemini quota is likely spent.')
    common.refresh_summary('demo', False)
    return 0


if __name__ == '__main__':
    sys.exit(main())
