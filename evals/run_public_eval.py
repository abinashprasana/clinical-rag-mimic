"""Golden set evaluation of the public Vercel runtime (demo/runtime.py), which
retrieves with Gemini embeddings and answers with Gemini. Fabricated notes
only; this module reads demo/runtime.py and never changes it.

    python -m evals.run_public_eval
    python -m evals.run_public_eval --pause 6 --max-fallbacks 3

Scores the reviewed questions: recall@5 and MRR over the passages the
runtime cites, answer correctness, refusal accuracy, and routing. The public
runtime has no FDA label tool and no clarify step of its own, so routing is
scored on what it does instead: a drug label question counts as handled when
it is not answered from a patient's note (the runtime's dosing notice or an
FDA route), and a one or two word question when the reply asks to clarify. The
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


def routed_correctly(behavior, result):
    if behavior == 'fda_lookup':
        return result.get('route') in ('dosage', 'direct') and not result.get('citations')
    return bool(result.get('needs_clarification')) or result.get('route') == 'clarify'


def run(questions, corpus, runner, pause, max_fallbacks, sleep=time.sleep):
    records, fallbacks_in_row = [], 0
    for q in questions:
        result = runner(q['question'])
        gemini = via_gemini(result)
        answer = result.get('final_answer') or ''
        keys = [corpus.keys[c['chunk_idx']] for c in result.get('citations', [])]
        record = {'id': q['id'], 'category': q['category'], 'behavior': q['expected_behavior'],
                  'gemini': gemini, 'route': result.get('route')}
        if q['expected_behavior'] == 'answer':
            gold = {(str(r['note_id']), r['section']) for r in q['relevant']}
            metrics = query_metrics(keys, gold)
            record['recall@5'], record['mrr'] = metrics['recall@5'], metrics['mrr']
            record['correct'] = score_answer(q, answer) and not is_refusal(answer)
        elif q['expected_behavior'] == 'refuse':
            record['correct'] = is_refusal(answer)
        else:
            record['correct'] = routed_correctly(q['expected_behavior'], result)
        records.append(record)
        if q['expected_behavior'] in ('fda_lookup', 'clarify'):
            sleep(pause)
            continue  # routing replies need not come from Gemini
        fallbacks_in_row = 0 if gemini or result.get('route') == 'direct' else fallbacks_in_row + 1
        if fallbacks_in_row >= max_fallbacks:
            break
        sleep(pause)
    return records


def summarise(records):
    routing = [r for r in records if r['behavior'] in ('fda_lookup', 'clarify')]
    # The dosing notice (route "direct") is the runtime's own answer, not a
    # quota fallback, so it is scored like a Gemini answer.
    scored = [r for r in records if (r['gemini'] or r.get('route') == 'direct') and r not in routing]
    answer = [r for r in scored if r['behavior'] == 'answer']
    refuse = [r for r in scored if r['behavior'] == 'refuse']

    def rate(rows):
        k, n = sum(r['correct'] for r in rows), len(rows)
        p, low, high = wilson_ci(k, n)
        return {'k': k, 'n': n, 'rate': p, 'low': low, 'high': high} if n else None

    def interval(metric):
        if not answer:
            return None
        mean, low, high = bootstrap_ci([r[metric] for r in answer])
        return {'mean': mean, 'low': low, 'high': high}

    return {
        'attempted': len(records),
        'answered_by_gemini': sum(r['gemini'] for r in scored),
        'scored': len(scored),
        'fallback_answers_excluded': len(records) - len(routing) - len(scored),
        'recall@5': interval('recall@5'),
        'mrr': interval('mrr'),
        'answer_correctness': rate(answer),
        'refusal_accuracy': rate(refuse),
        'routing_accuracy': rate(routing),
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
]
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
        ['MRR (cited passages)', 'n/a' if not summary['mrr'] else common.fmt_ci(
            summary['mrr']['mean'], summary['mrr']['low'], summary['mrr']['high'])],
        ['answer correctness', pct(summary['answer_correctness'])],
        ['refusal accuracy', pct(summary['refusal_accuracy'])],
        ['routing (drug label or clarify)', pct(summary['routing_accuracy'])],
    ]))
    if len(records) < len(questions):
        print(f'Stopped after {len(records)} of {len(questions)} questions: '
              f'{args.max_fallbacks} fallbacks in a row, so the Gemini quota is likely spent.')
    common.refresh_summary('demo', False)
    return 0


if __name__ == '__main__':
    sys.exit(main())
