"""Terminal tool for labelling generated answers against their passages.

    python -m evals.label_tool --corpus demo                 # label, resumes where you stopped
    python -m evals.label_tool --corpus demo --relabel 20    # second pass on 20 random items, another day
    python -m evals.label_tool --corpus real                 # reads and writes outputs/eval/ only

Run python -m evals.run_generation_eval first; it writes the records this
tool reads. Each item is the first draft the faithfulness gate judged,
shown with the question and the retrieved passages. Labels are
  s  supported            every claim is backed by the passages
  p  partially supported  some claims are backed, some are not
  u  unsupported          the main claim is not backed
The gate's own decision is never shown. Labels store the question id, a
hash of the draft, the label, the round and a timestamp; never the text.
"""
import argparse
import datetime
import hashlib
import json
import os
import random
import sys

from evals.common import DEMO_RUNS_DIR, REAL_DIR
from evals.corpus import load_corpus
from evals.golden import GOLDEN_FILES, load_questions

LABELS = {'s': 'supported', 'p': 'partially_supported', 'u': 'unsupported'}
RELABEL_SEED = 7
PATHS = {
    'demo': {'records': os.path.join(DEMO_RUNS_DIR, 'generation_records_demo.jsonl'),
             'labels': os.path.join('evals', 'labels', 'demo_labels.jsonl')},
    'real': {'records': os.path.join(REAL_DIR, 'generation_records_real.jsonl'),
             'labels': os.path.join(REAL_DIR, 'labels_real.jsonl')},
}


def draft_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def item_key(record):
    return f"{record['id']}:{draft_hash(record['first_draft'])}"


def labellable(records):
    """Records where the gate judged a draft (retrieval route, non empty)."""
    return [r for r in records if r.get('first_draft') and r.get('first_supported') is not None]


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def append_label(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(row) + '\n')


def pending(records, labels, round_no, relabel_n=None, seed=RELABEL_SEED):
    """Items still to label in this round."""
    done = {(row['key'], row['round']) for row in labels}
    items = labellable(records)
    if round_no == 1:
        return [r for r in items if (item_key(r), 1) not in done]
    first = {row['key'] for row in labels if row['round'] == 1}
    eligible = sorted((r for r in items if item_key(r) in first), key=item_key)
    chosen = random.Random(seed).sample(eligible, min(relabel_n or 20, len(eligible)))
    return [r for r in chosen if (item_key(r), 2) not in done]


def show(record, question, passages, out):
    out('=' * 78)
    out(f"{record['id']}  ({record['category']})")
    out(f'Question: {question}')
    for i, text in enumerate(passages, 1):
        out(f'--- passage {i} ---')
        out(text)
    out('--- generated answer ---')
    out(record['first_draft'])


def label_loop(items, questions, corpus, labels_path, round_no, ask=input, out=print, now=None):
    """Returns the number of labels written. `ask` and `out` are injectable for tests."""
    written = 0
    for n, record in enumerate(items, 1):
        out(f'\n[{n}/{len(items)}] round {round_no}')
        show(record, questions[record['id']], [corpus.chunks[i] for i in record['passages']], out)
        while True:
            choice = ask('Label [s]upported, [p]artially, [u]nsupported, [k] skip, [q] quit: ').strip().lower()
            if choice in LABELS or choice in ('k', 'q'):
                break
        if choice == 'q':
            break
        if choice == 'k':
            continue
        append_label(labels_path, {
            'key': item_key(record), 'id': record['id'], 'label': LABELS[choice], 'round': round_no,
            'labelled_at': (now or datetime.datetime.now)().isoformat(timespec='seconds'),
        })
        written += 1
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--corpus', choices=['demo', 'real'], default='demo')
    parser.add_argument('--relabel', type=int, metavar='N', help='second labelling round on N random items')
    args = parser.parse_args(argv)

    paths = PATHS[args.corpus]
    records = read_jsonl(paths['records'])
    if not records:
        print(f"No generation records at {paths['records']}. "
              f'Run python -m evals.run_generation_eval --corpus {args.corpus} first.')
        return 2
    labels = read_jsonl(paths['labels'])
    round_no = 2 if args.relabel else 1
    if round_no == 2:
        today = datetime.datetime.now().astimezone().date().isoformat()
        if any(row['labelled_at'].startswith(today) for row in labels if row['round'] == 1):
            print('Note: some first round labels are from today. The relabel pass is meant for another day.')
    items = pending(records, labels, round_no, args.relabel)
    if not items:
        print('Nothing left to label in this round.')
        return 0
    questions = {q['id']: q['question'] for q in load_questions(GOLDEN_FILES[args.corpus])}
    written = label_loop(items, questions, load_corpus(args.corpus), paths['labels'], round_no)
    print(f"\nSaved {written} labels to {paths['labels']}.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
