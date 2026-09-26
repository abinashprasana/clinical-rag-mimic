"""Retrieval evaluation: five variants side by side, overall and per category,
with bootstrap intervals, a paired difference against `dense`, and latency.

    python -m evals.run_retrieval_eval --corpus demo
    python -m evals.run_retrieval_eval --corpus demo --include-unreviewed   # development only
    python -m evals.run_retrieval_eval --corpus real                        # local, writes outputs/eval/

Only questions with expected_behavior "answer" carry retrieval labels.
"""
import argparse
import os
import sys
import time

from evals import common
from evals.corpus import load_corpus
from evals.golden import GOLDEN_FILES, load_questions, select, validate
from evals.metrics import bootstrap_ci, paired_bootstrap_diff, percentile, query_metrics
from evals.retrievers import DEFAULT_VARIANT, RERANKER_MODEL, VARIANTS, build_retrievers

HEADLINE = ('recall@1', 'recall@3', 'recall@5', 'mrr', 'ndcg@5')
PAIRED = ('recall@5', 'mrr')


def gold_keys(question):
    return {(str(r['note_id']), r['section']) for r in question['relevant']}


def evaluate_variant(retriever, corpus, questions, k=5):
    """Per question records for one variant. The first question is run once
    untimed as a warm up so model loading does not land in latency."""
    if questions:
        retriever.retrieve(questions[0]['question'], k=k)
    records = []
    for q in questions:
        start = time.perf_counter()
        indices = retriever.retrieve(q['question'], k=k)
        latency_ms = (time.perf_counter() - start) * 1000
        keys = [corpus.keys[i] for i in indices]
        records.append({
            'id': q['id'],
            'category': q['category'],
            'retrieved': [list(key) for key in keys],
            'latency_ms': latency_ms,
            **query_metrics(keys, gold_keys(q)),
        })
    return records


def aggregate(records_by_variant, baseline=DEFAULT_VARIANT):
    """{variant: {metric: {mean, low, high}, latency_p50_ms, latency_p95_ms,
    paired_vs_dense: {metric: {mean, low, high}}, n}} for one question subset."""
    out = {}
    base = records_by_variant.get(baseline)
    for variant, records in records_by_variant.items():
        row = {'n': len(records)}
        for metric in HEADLINE + ('hit@5', 'note_hit@5'):
            mean, low, high = bootstrap_ci([r[metric] for r in records])
            row[metric] = {'mean': mean, 'low': low, 'high': high}
        latencies = [r['latency_ms'] for r in records]
        row['latency_p50_ms'] = percentile(latencies, 50)
        row['latency_p95_ms'] = percentile(latencies, 95)
        if base is not None and variant != baseline:
            row['paired_vs_dense'] = {}
            for metric in PAIRED:
                mean, low, high = paired_bootstrap_diff(
                    [r[metric] for r in records], [r[metric] for r in base])
                row['paired_vs_dense'][metric] = {'mean': mean, 'low': low, 'high': high}
        out[variant] = row
    return out


def by_category(records_by_variant):
    categories = sorted({r['category'] for recs in records_by_variant.values() for r in recs})
    return {
        category: aggregate({
            v: [r for r in recs if r['category'] == category] for v, recs in records_by_variant.items()
        })
        for category in categories
    }


def table_rows(agg):
    rows = []
    for variant, row in agg.items():
        cells = [variant, row['n']]
        cells += [common.fmt_ci(row[m]['mean'], row[m]['low'], row[m]['high']) for m in HEADLINE]
        paired = row.get('paired_vs_dense', {}).get('recall@5')
        cells.append(common.fmt_ci(paired['mean'], paired['low'], paired['high']) if paired else 'baseline')
        cells.append(f"{row['note_hit@5']['mean']:.3f}")
        cells += [f'{row["latency_p50_ms"]:.1f}', f'{row["latency_p95_ms"]:.1f}']
        rows.append(cells)
    return rows


HEADERS = ['variant', 'n', 'recall@1', 'recall@3', 'recall@5', 'MRR', 'nDCG@5',
           'recall@5 minus dense (paired)', 'note hit@5', 'p50 ms', 'p95 ms']


def render(result, fmt):
    table = common.markdown_table if fmt == 'md' else common.text_table
    parts = []
    if not result['meta']['reviewed_only']:
        parts.append(common.UNREVIEWED_BANNER)
    parts.append(f'Corpus {result["meta"]["corpus"]}, {result["meta"]["n_questions"]} retrieval questions, '
                 f'top 5, mean with 95% bootstrap interval ({result["bootstrap"]}).')
    if result['meta']['corpus'] == 'demo':
        parts.append(common.DEMO_SIZE_NOTE)
    parts.append('Overall\n' + table(HEADERS, table_rows(result['overall'])))
    for category, agg in result['by_category'].items():
        parts.append(f'Category: {category}\n' + table(HEADERS, table_rows(agg)))
    return '\n\n'.join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--corpus', choices=['demo', 'real'], default='demo')
    parser.add_argument('--include-unreviewed', action='store_true',
                        help='also score draft questions (development only, labelled UNREVIEWED)')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--file', help='override the golden file path')
    args = parser.parse_args(argv)
    common.guard_real_data(args.corpus)

    path = args.file or GOLDEN_FILES[args.corpus]
    if not os.path.exists(path):
        print(f'{path} does not exist. See evals/golden/README_real_questions.md.')
        return 2
    corpus = load_corpus(args.corpus)
    all_questions = load_questions(path)
    errors, _ = validate(all_questions, corpus)
    if errors:
        print(f'{len(errors)} validation errors; run python -m evals.golden --corpus {args.corpus}')
        return 1
    questions = [q for q in select(all_questions, args.include_unreviewed)
                 if q['expected_behavior'] == 'answer']
    if not questions:
        print('No reviewed retrieval questions. Mark questions "reviewed": true, '
              'or pass --include-unreviewed for a development run.')
        return 2

    retrievers = build_retrievers(corpus, args.variants)
    records_by_variant = {v: evaluate_variant(retrievers[v], corpus, questions) for v in args.variants}

    result = {
        'meta': common.run_metadata(args.corpus, args.include_unreviewed, len(questions)),
        'bootstrap': '1000 resamples, fixed seed',
        'reranker': RERANKER_MODEL,
        'overall': aggregate(records_by_variant),
        'by_category': by_category(records_by_variant),
    }
    agg_dir, records_dir = common.output_dirs(args.corpus, args.include_unreviewed)
    stem = f'retrieval_{args.corpus}' + ('.unreviewed' if args.include_unreviewed else '')
    common.write_json(os.path.join(agg_dir, stem + '.json'), result)
    with open(os.path.join(agg_dir, stem + '.md'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(render(result, 'md') + '\n')
    common.write_jsonl(
        os.path.join(records_dir, f'retrieval_records_{args.corpus}.jsonl'),
        [{'variant': v, **r} for v, recs in records_by_variant.items() for r in recs],
    )
    print(render(result, 'text'))
    common.refresh_summary(args.corpus, args.include_unreviewed)
    print(f'\nWrote {agg_dir}/{stem}.json and .md; per question records in {records_dir}/.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
