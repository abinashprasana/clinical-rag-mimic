"""How well the faithfulness gate agrees with human labels, next to the
labeller's agreement with themself, plus the synthetic stress set.

    python -m evals.faithfulness_agreement --corpus demo
    python -m evals.faithfulness_agreement --corpus demo --stress-source gate-passed   # before labels exist

Binary mapping: a human label of "supported" counts as supported; "partially
supported" and "unsupported" both count as not supported.
* agreement           share of items where the gate's first decision matches the binary label
* kappa               Cohen's kappa on the same binary pairs
* false pass rate     not supported answers the gate passed / all not supported answers
* false refusal rate  supported answers the gate refused / all supported answers
"""
import argparse
import os
import sys

from evals import common
from evals.corpus import load_corpus
from evals.golden import GOLDEN_FILES, load_questions
from evals.label_tool import PATHS, item_key, labellable, read_jsonl
from evals.metrics import cohen_kappa, confusion_matrix, wilson_ci
from evals.stress import CORRUPTIONS, run_stress

MAPPING = 'supported = supported; partially supported and unsupported = not supported'


def _rate(k, n):
    p, low, high = wilson_ci(k, n)
    return {'k': k, 'n': n, 'rate': p, 'low': low, 'high': high}


def agreement(records, labels):
    """Gate versus round 1 human labels on items whose draft is unchanged."""
    by_key = {item_key(r): r for r in labellable(records)}
    pairs = [(row, by_key[row['key']]) for row in labels if row['round'] == 1 and row['key'] in by_key]
    stale = sum(1 for row in labels if row['round'] == 1 and row['key'] not in by_key)
    if not pairs:
        return {'n': 0, 'stale_labels': stale}
    human = ['supported' if row['label'] == 'supported' else 'not_supported' for row, _ in pairs]
    gate = ['supported' if rec['first_supported'] else 'not_supported' for _, rec in pairs]
    not_sup = [(h, rec) for h, (_, rec) in zip(human, pairs) if h == 'not_supported']
    sup = [(h, rec) for h, (_, rec) in zip(human, pairs) if h == 'supported']
    return {
        'n': len(pairs),
        'stale_labels': stale,
        'mapping': MAPPING,
        'agreement': _rate(sum(h == g for h, g in zip(human, gate)), len(pairs)),
        'kappa': cohen_kappa(human, gate),
        'false_pass': _rate(sum(rec['first_supported'] for _, rec in not_sup), len(not_sup)),
        'false_refusal': _rate(sum(rec['outcome'] == 'refused_by_gate' for _, rec in sup), len(sup)),
        'confusion': confusion_matrix(human, gate, ['supported', 'not_supported']),
    }


def self_agreement(labels):
    first = {row['key']: row['label'] for row in labels if row['round'] == 1}
    second = {row['key']: row['label'] for row in labels if row['round'] == 2}
    keys = sorted(first.keys() & second.keys())
    if not keys:
        return {'n': 0}
    a, b = [first[k] for k in keys], [second[k] for k in keys]
    binary = lambda xs: ['supported' if x == 'supported' else 'not_supported' for x in xs]
    return {
        'n': len(keys),
        'agreement_3way': _rate(sum(x == y for x, y in zip(a, b)), len(keys)),
        'agreement_binary': _rate(sum(x == y for x, y in zip(binary(a), binary(b))), len(keys)),
        'kappa_binary': cohen_kappa(binary(a), binary(b)),
    }


def stress_items(records, labels, corpus, source):
    if source == 'human-supported':
        keys = {row['key'] for row in labels if row['round'] == 1 and row['label'] == 'supported'}
        chosen = [r for r in labellable(records) if item_key(r) in keys]
    else:
        chosen = [r for r in labellable(records) if r['first_supported']]
    return [(r['first_draft'], [corpus.chunks[i] for i in r['passages']]) for r in chosen]


def fmt(row):
    if not row or not row.get('n'):
        return 'n/a'
    return f"{row['k']}/{row['n']} = {row['rate']:.0%} [{row['low']:.0%}, {row['high']:.0%}]"


def render(result, fmt_name):
    table = common.markdown_table if fmt_name == 'md' else common.text_table
    parts = []
    if not result['meta']['reviewed_only']:
        parts.append(common.UNREVIEWED_BANNER)
    a, s = result['agreement'], result['self_agreement']
    if a['n']:
        parts.append(f'Binary mapping: {MAPPING}.')
        parts.append(table(['gate versus human labels', 'value'], [
            ['items', a['n']],
            ['agreement', fmt(a['agreement'])],
            ['Cohen kappa', 'n/a' if a['kappa'] != a['kappa'] else f"{a['kappa']:.2f}"],
            ['false pass rate', fmt(a['false_pass'])],
            ['false refusal rate', fmt(a['false_refusal'])],
            ['self agreement, binary (relabelled items)', fmt(s.get('agreement_binary'))],
            ['self agreement, three way', fmt(s.get('agreement_3way'))],
        ]))
        c = a['confusion']
        parts.append(table(['human \\ gate', 'passed', 'flagged'], [
            ['supported', c['supported']['supported'], c['supported']['not_supported']],
            ['not supported', c['not_supported']['supported'], c['not_supported']['not_supported']],
        ]))
    else:
        parts.append('No human labels match the current generation records yet. '
                     'Label with python -m evals.label_tool.')
    if a.get('stale_labels'):
        parts.append(f"{a['stale_labels']} labels refer to drafts that changed since labelling and were skipped.")
    st = result['stress']
    parts.append(f"SYNTHETIC CORRUPTION (never blended with the human label numbers). Source answers: {st['source']}.")
    parts.append(table(['corruption', 'detected by the gate'],
                       [[name, fmt(_rate(st['counts'][name]['detected'], st['counts'][name]['n']))]
                        for name in CORRUPTIONS]))
    return '\n\n'.join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--corpus', choices=['demo', 'real'], default='demo')
    parser.add_argument('--include-unreviewed', action='store_true')
    parser.add_argument('--stress-source', choices=['human-supported', 'gate-passed'], default='human-supported',
                        help='gate-passed uses drafts the gate passed, for development before labels exist')
    args = parser.parse_args(argv)
    common.guard_real_data(args.corpus)

    records = read_jsonl(PATHS[args.corpus]['records'])
    if not records:
        print('No generation records. Run python -m evals.run_generation_eval first.')
        return 2
    questions = load_questions(GOLDEN_FILES[args.corpus])
    allowed = {q['id'] for q in questions if args.include_unreviewed or q['reviewed']}
    records = [r for r in records if r['id'] in allowed]
    labels = [row for row in read_jsonl(PATHS[args.corpus]['labels']) if row['id'] in allowed]
    corpus = load_corpus(args.corpus)

    source_note = {'human-supported': 'answers labelled supported by the human labeller',
                   'gate-passed': 'drafts the gate passed (no human labels used; development only)'}
    result = {
        'meta': common.run_metadata(args.corpus, args.include_unreviewed, len(records)),
        'agreement': agreement(records, labels),
        'self_agreement': self_agreement(labels),
        'stress': {'source': source_note[args.stress_source],
                   'counts': run_stress(stress_items(records, labels, corpus, args.stress_source))},
    }
    agg_dir, _ = common.output_dirs(args.corpus, args.include_unreviewed or args.stress_source == 'gate-passed')
    stem = f'faithfulness_{args.corpus}' + (
        '.unreviewed' if args.include_unreviewed or args.stress_source == 'gate-passed' else '')
    common.write_json(os.path.join(agg_dir, stem + '.json'), result)
    with open(os.path.join(agg_dir, stem + '.md'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(render(result, 'md') + '\n')
    print(render(result, 'text'))
    print(f'\nWrote {agg_dir}/{stem}.json and .md.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
