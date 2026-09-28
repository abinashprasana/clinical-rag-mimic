"""Golden question sets: loader, schema validator and category report.

    python -m evals.golden --corpus demo
    python -m evals.golden --corpus real      # reads evals/golden/real_questions.local.jsonl

For the real corpus the report prints question ids, counts and error codes
only, never question text, so its console output is safe to share.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

from evals.corpus import canonical_section, load_corpus

GOLDEN_DIR = os.path.join('evals', 'golden')
GOLDEN_FILES = {
    'demo': os.path.join(GOLDEN_DIR, 'demo_questions.jsonl'),
    'real': os.path.join(GOLDEN_DIR, 'real_questions.local.jsonl'),
}

BEHAVIORS = ('answer', 'refuse', 'clarify', 'fda_lookup')
# Category -> the behaviour a question in it must expect.
CATEGORIES = {
    'single_fact': 'answer',
    'medication_list': 'answer',
    'multi_section': 'answer',
    'negation': 'answer',
    'discharge_followup': 'answer',
    'unanswerable': 'refuse',
    'fda_dosage': 'fda_lookup',
    'ambiguous': 'clarify',
}
DEMO_TARGET_COUNTS = {
    'single_fact': 24, 'medication_list': 16, 'multi_section': 12, 'negation': 12,
    'unanswerable': 16, 'fda_dosage': 8, 'ambiguous': 6, 'discharge_followup': 6,
}
FIELDS = {
    'id': str, 'category': str, 'question': str, 'expected_behavior': str,
    'relevant': list, 'must_contain': list, 'must_not_contain': list,
    'reviewed': bool, 'notes': str,
}
# scope_note_id: the admission (hadm_id) a question is about, so a note
# scoped run answers it from that note only. Optional.
OPTIONAL_FIELDS = {'scope_note_id': str}
_ID_RE = re.compile(r'^q\d{3}$')
_DASHES = dict.fromkeys(map(ord, '‐‑‒–—―−'), '-')


def normalise(text):
    """Lowercase, unicode dashes to ASCII, whitespace collapsed. Numbers and
    units are left exactly as written, so "40 mg" never matches "4 mg"."""
    return re.sub(r'\s+', ' ', text.translate(_DASHES).lower()).strip()


def contains(text, phrase):
    return normalise(phrase) in normalise(text)


def load_questions(path):
    questions = []
    with open(path, encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    questions.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{line_no}: invalid JSON ({exc.msg})') from exc
    return questions


SPLIT_FILE = os.path.join(GOLDEN_DIR, 'demo_split.json')


def select(questions, include_unreviewed=False, split='all'):
    """Reviewed questions (or all with include_unreviewed), optionally limited
    to the frozen dev or test half of the demo set."""
    chosen = [q for q in questions if include_unreviewed or q.get('reviewed') is True]
    if split == 'all':
        return chosen
    with open(SPLIT_FILE, encoding='utf-8') as f:
        ids = set(json.load(f)[split])
    return [q for q in chosen if q['id'] in ids]


def validate(questions, corpus=None):
    """Returns (errors, warnings), each a list of (question id, message).
    Messages name fields and ids, never question or note text."""
    errors, warnings = [], []
    seen, seen_text = set(), set()
    for n, q in enumerate(questions, 1):
        qid = q.get('id', f'line {n}')
        for field, kind in FIELDS.items():
            if field not in q:
                errors.append((qid, f'missing field {field}'))
            elif not isinstance(q[field], kind):
                errors.append((qid, f'{field} must be {kind.__name__}'))
        for field in sorted(set(q) - set(FIELDS) - set(OPTIONAL_FIELDS)):
            errors.append((qid, f'unknown field {field}'))
        scope = q.get('scope_note_id')
        if scope is not None:
            if not isinstance(scope, str):
                errors.append((qid, 'scope_note_id must be str'))
            elif corpus is not None and scope not in corpus.sections_by_note():
                errors.append((qid, f'scope_note_id {scope} is not in the index'))
        if not isinstance(q.get('id'), str) or not _ID_RE.match(q['id']):
            errors.append((qid, 'id must look like q001'))
        elif qid in seen:
            errors.append((qid, 'duplicate id'))
        seen.add(qid)
        text = q.get('question')
        if isinstance(text, str) and text.strip().lower() in seen_text:
            errors.append((qid, 'duplicate question text'))
        if isinstance(text, str):
            seen_text.add(text.strip().lower())

        behavior, category = q.get('expected_behavior'), q.get('category')
        if behavior not in BEHAVIORS:
            errors.append((qid, f'expected_behavior must be one of {BEHAVIORS}'))
        if category not in CATEGORIES:
            errors.append((qid, f'category must be one of {tuple(CATEGORIES)}'))
        elif behavior in BEHAVIORS and CATEGORIES[category] != behavior:
            errors.append((qid, f'category {category} expects behaviour {CATEGORIES[category]}'))

        relevant = q.get('relevant') if isinstance(q.get('relevant'), list) else []
        if behavior == 'answer' and not relevant:
            errors.append((qid, 'answer questions need at least one relevant entry'))
        if behavior in ('refuse', 'clarify', 'fda_lookup') and relevant:
            errors.append((qid, f'{behavior} questions must have an empty relevant list'))
        if behavior == 'answer' and not q.get('must_contain') and not q.get('must_not_contain'):
            warnings.append((qid, 'answer question has no must_contain or must_not_contain'))

        valid_entries = []
        for entry in relevant:
            if not isinstance(entry, dict) or set(entry) != {'note_id', 'section'}:
                errors.append((qid, 'relevant entries need exactly note_id and section'))
            elif canonical_section(entry['section']) is None:
                errors.append((qid, f'section {entry["section"]!r} is not a MIMIC section header'))
            elif corpus is not None and not corpus.has_entry(entry['note_id'], entry['section']):
                errors.append((qid, f'note {entry["note_id"]} has no indexed {entry["section"]} chunk'))
            else:
                valid_entries.append(entry)

        # must_contain terms should be findable in the gold passages; a miss
        # usually means a typo in the gold entry rather than a model failure.
        if corpus is not None and behavior == 'answer' and valid_entries:
            gold_text = ' '.join(
                corpus.chunks[i]
                for entry in valid_entries
                for i in corpus.chunk_indices(entry['note_id'], entry['section'])
            )
            for k, term in enumerate(q.get('must_contain') or []):
                if isinstance(term, str) and not contains(gold_text, term):
                    warnings.append((qid, f'must_contain[{k}] not found in the gold passages'))
    return errors, warnings


def report(questions, corpus_name):
    counts = Counter(q.get('category') for q in questions)
    reviewed = Counter(q.get('category') for q in questions if q.get('reviewed') is True)
    header = f'{"category":<20} {"total":>5} {"reviewed":>8}'
    lines = [header + ('  target' if corpus_name == 'demo' else '')]
    for category in CATEGORIES:
        row = f'{category:<20} {counts[category]:>5} {reviewed[category]:>8}'
        if corpus_name == 'demo':
            row += f'  {DEMO_TARGET_COUNTS[category]:>6}'
        lines.append(row)
    lines.append(f'{"all":<20} {sum(counts.values()):>5} {sum(reviewed.values()):>8}')
    retrieval = sum(1 for q in questions if q.get('expected_behavior') == 'answer')
    lines.append(f'{retrieval} questions carry retrieval labels (expected_behavior = answer).')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Validate a golden question set.')
    parser.add_argument('--corpus', choices=['demo', 'real'], default='demo')
    parser.add_argument('--file', help='override the golden file path')
    args = parser.parse_args(argv)

    path = args.file or GOLDEN_FILES[args.corpus]
    if not os.path.exists(path):
        print(f'{path} does not exist. See evals/golden/README_real_questions.md.')
        return 2
    questions = load_questions(path)
    corpus = load_corpus(args.corpus)
    errors, warnings = validate(questions, corpus)

    print(f'{path}: {len(questions)} questions, corpus {args.corpus} '
          f'({len(corpus.sections_by_note())} notes, {len(corpus.chunks)} chunks)\n')
    print(report(questions, args.corpus))
    for label, items in (('ERROR', errors), ('WARNING', warnings)):
        for qid, message in items:
            print(f'{label} {qid}: {message}')
    print(f'\n{len(errors)} errors, {len(warnings)} warnings.')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
