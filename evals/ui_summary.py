"""Aggregate evaluation numbers for the web UI, in one small fixed schema.

app.py imports this module in both runtimes, including the public Vercel
function, so it uses only the standard library. build_summary() runs after
each reviewed evaluation run; load_summary() returns None for a missing or
malformed file, and the UI then shows a pending state.

The summary holds aggregates only: rates, intervals and counts. It never
holds question text, answers, passages or question ids.
"""
import json
import math
import os

SCHEMA = 1
DEMO_SUMMARY = os.path.join('evals', 'results', 'summary_demo.json')
REAL_SUMMARY = os.path.join('outputs', 'eval', 'summary_real.json')
SECTIONS = ('retrieval', 'generation', 'faithfulness', 'stress', 'injection', 'public')


def _read(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _interval(row):
    return {'mean': row['mean'], 'low': row['low'], 'high': row['high']}


def _wilson(k, n, z=1.959964):
    """95% Wilson interval (standard library only, as this module is)."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _rate(row):
    return {k: row[k] for k in ('k', 'n', 'rate', 'low', 'high')} if row and row.get('n') else None


def build_summary(results_dir, corpus):
    """Combine the reviewed aggregate files in results_dir into one summary.
    Unreviewed results never live in results_dir, so they never reach it."""
    # The demo headline comes from the 40 question holdout, which was never
    # used for tuning; the 100 original questions were tuning data.
    def reported(kind):
        holdout = _read(os.path.join(results_dir, f'{kind}_{corpus}_demo_holdout.json'))
        return holdout or _read(os.path.join(results_dir, f'{kind}_{corpus}.json'))

    retrieval, generation = reported('retrieval'), reported('generation')
    faithfulness = _read(os.path.join(results_dir, f'faithfulness_{corpus}.json'))
    injection = _read(os.path.join(results_dir, f'injection_{corpus}.json')) if corpus == 'demo' else None
    summary = {'schema': SCHEMA, 'corpus': corpus, **dict.fromkeys(SECTIONS)}
    summary['question_set'] = ('held-out' if os.path.exists(
        os.path.join(results_dir, f'generation_{corpus}_demo_holdout.json')) else 'all')
    public = _read(os.path.join(results_dir, 'public_demo_demo_holdout.json')) if corpus == 'demo' else None

    if retrieval and retrieval['meta']['reviewed_only']:
        variants = []
        for name, row in retrieval['overall'].items():
            diff = row.get('paired_vs_dense', {}).get('recall@5')
            variants.append({
                'name': name,
                'recall@5': _interval(row['recall@5']), 'mrr': _interval(row['mrr']),
                'ndcg@5': _interval(row['ndcg@5']),
                'diff_recall@5': _interval(diff) if diff else None,
                'p50_ms': row['latency_p50_ms'], 'p95_ms': row['latency_p95_ms'],
            })
        summary['retrieval'] = {'n': retrieval['meta']['n_questions'], 'default': 'dense',
                                'run_at': retrieval['meta']['run_at'], 'variants': variants}

    if generation and generation['meta']['reviewed_only']:
        h = generation['scores']['headline']
        summary['generation'] = {
            'n': generation['meta']['n_questions'], 'run_at': generation['meta']['run_at'],
            'answer_correctness': _rate(h['answer_correctness']),
            'refusal_accuracy': _rate(h['refusal_accuracy']),
            'routing_accuracy': _rate(h['routing_accuracy']),
        }

    if faithfulness and faithfulness['meta']['reviewed_only']:
        a, s = faithfulness['agreement'], faithfulness['self_agreement']
        if a.get('n'):
            kappa = a['kappa']
            summary['faithfulness'] = {
                'n': a['n'], 'agreement': _rate(a['agreement']),
                'kappa': None if math.isnan(kappa) else kappa,
                'false_pass': _rate(a['false_pass']), 'false_refusal': _rate(a['false_refusal']),
                'self_agreement': _rate(s.get('agreement_binary')),
            }
        # Synthetic corruptions, reported on their own and never blended with
        # the human label numbers. The source says which answers were broken.
        stress = faithfulness['stress']
        rows = []
        for name, v in stress['counts'].items():
            if v['n']:
                low, high = _wilson(v['detected'], v['n'])
                rows.append({'name': name, 'k': v['detected'], 'n': v['n'],
                             'rate': v['detected'] / v['n'], 'low': low, 'high': high})
        if rows:
            source = 'human-labelled' if stress['source'].startswith('answers labelled supported') else 'gate-passed'
            summary['stress'] = {'rows': rows, 'source': source, 'run_at': faithfulness['meta']['run_at']}

    public = public or (_read(os.path.join(results_dir, 'public_demo.json')) if corpus == 'demo' else None)
    if public and public.get('answered_by_gemini'):
        summary['public'] = {
            'run_at': public['meta']['run_at'], 'attempted': public['attempted'],
            'answered_by_gemini': public['answered_by_gemini'],
            'recall@5': _interval(public['recall@5']) if public.get('recall@5') else None,
            'mrr': _interval(public['mrr']) if public.get('mrr') else None,
            'answer_correctness': _rate(public.get('answer_correctness')),
            'refusal_accuracy': _rate(public.get('refusal_accuracy')),
            'routing_accuracy': _rate(public.get('routing_accuracy')),
        }

    if injection:
        paths = []
        for name, p in injection['paths'].items():
            if p:
                paths.append({'name': name,
                              'canary_final': _rate(p['canary_in_final_answer']),
                              'gate_passed': _rate(p['gate_passed_canary_draft'])})
        summary['injection'] = {'questions': injection['n_questions'], 'run_at': injection['meta']['run_at'],
                                'paths': paths}
    return summary


def _is_prob(x):
    return isinstance(x, (int, float)) and 0 <= x <= 1


def _valid_rate(row):
    return row is None or (
        isinstance(row, dict) and isinstance(row.get('k'), int) and isinstance(row.get('n'), int)
        and all(_is_prob(row.get(f)) for f in ('rate', 'low', 'high')))


def valid(summary):
    if not isinstance(summary, dict) or summary.get('schema') != SCHEMA:
        return False
    if set(summary) - {'schema', 'corpus', 'question_set', *SECTIONS}:
        return False
    r = summary.get('retrieval')
    if r is not None and not all(
            _is_prob(v[m][f]) for v in r.get('variants', []) for m in ('recall@5', 'mrr', 'ndcg@5')
            for f in ('mean', 'low', 'high')):
        return False
    g = summary.get('generation')
    if g is not None and not all(_valid_rate(g.get(k)) for k in
                                 ('answer_correctness', 'refusal_accuracy', 'routing_accuracy')):
        return False
    f = summary.get('faithfulness')
    if f is not None and not all(_valid_rate(f.get(k)) for k in
                                 ('agreement', 'false_pass', 'false_refusal', 'self_agreement')):
        return False
    s = summary.get('stress')
    if s is not None and not (isinstance(s.get('rows'), list) and all(_valid_rate(r) for r in s['rows'])):
        return False
    p = summary.get('public')
    return p is None or all(_valid_rate(p.get(k)) for k in ('answer_correctness', 'refusal_accuracy'))


def load_summary(path):
    """The summary as a dict, or None when missing, unreadable or malformed."""
    summary = _read(path)
    return summary if valid(summary) else None


def write_summary(results_dir, corpus, path):
    summary = build_summary(results_dir, corpus)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(summary, f, indent=2)
        f.write('\n')
    return summary


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Rebuild the UI summary from reviewed results.')
    parser.add_argument('--corpus', choices=['demo', 'real'], default='demo')
    args = parser.parse_args(argv)
    results_dir, path = (('evals/results', DEMO_SUMMARY) if args.corpus == 'demo'
                         else ('outputs/eval', REAL_SUMMARY))
    summary = write_summary(results_dir, args.corpus, path)
    filled = [s for s in SECTIONS if summary[s]]
    print(f'Wrote {path}; sections with results: {", ".join(filled) or "none yet"}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
