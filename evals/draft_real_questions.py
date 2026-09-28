"""Drafts golden questions from a local index, entirely on this machine.

    python -m evals.draft_real_questions --corpus real               # writes evals/golden/real_questions.local.jsonl
    python -m evals.draft_real_questions --corpus demo --show 10     # try it on the fabricated notes

Real MIMIC-IV-Note text must not be sent to any online service, so the
questions for the real set are drafted here by fixed rules and written to a
gitignored *.local.* file. The console gets category counts only for the
real corpus; --show refuses to print real questions.

Every draft is reviewed=false. The rules pull a fact straight from one
section of one note (a lab value, the first discharge medications, a
"denies X" finding, a test the note never mentions, a follow up specialty),
so a draft is usually right, but it is still a draft: read each one against
the note before setting reviewed=true. The patient is named by a phrase
from that note (chief complaint, the "presents with" phrase of the history,
or the discharge diagnosis) that no other note in the index shares.
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

from evals.corpus import load_corpus
from evals.golden import CATEGORIES, GOLDEN_FILES

REAL_TARGETS = {'single_fact': 24, 'medication_list': 16, 'multi_section': 12, 'negation': 12,
                'unanswerable': 16, 'fda_dosage': 8, 'ambiguous': 6, 'discharge_followup': 6}
NOTE = 'Drafted locally by evals/draft_real_questions.py; check against the note before setting reviewed.'

_HEADER = re.compile(r'^\[[^\]]+\]\s*')
_LABS = {  # MIMIC style "WBC-7.2" and plain "WBC 14.2" / "hemoglobin 12.8"
    'white blood cell count': r'\b(?:WBC)[-:\s]+(\d+(?:\.\d+)?)',
    'hemoglobin': r'\b(?:Hgb|hemoglobin)[-:\s]+(\d+(?:\.\d+)?)',
    'creatinine': r'\b(?:Creat|creatinine)[-:\s]+(\d+(?:\.\d+)?)',
    'sodium': r'\b(?:Na|sodium)[-:\s]+(\d{3})\b',
    'potassium': r'\b(?:K|potassium)[-:\s]+(\d(?:\.\d)?)\b',
    'glucose': r'\b(?:Glucose|glucose)[-:\s]+(\d{2,3})\b',
    'platelet count': r'\b(?:Plt|platelets)[-:\s]+(\d{2,4})\b',
    'INR': r'\b(?:INR\(PT\)|INR)[-:\s]+(\d+(?:\.\d+)?)',
}
_BP = re.compile(r'\bBP[:\s]+(\d{2,3}/\d{2,3})')
_MED = re.compile(r'(?:^|\s)\d{1,2}\.\s+([A-Za-z][A-Za-z-]{2,})')
_NEG = re.compile(r'\b(denies|denied|no|without)\s+((?:[a-z]{3,}\s)?[a-z]{3,})\b', re.IGNORECASE)
_FOLLOW = re.compile(r'follow(?:\s|-)?up with (?:your |the |a |an |dr\.? )?([a-z]+(?: [a-z]+)?)', re.IGNORECASE)
_SPECIALTIES = ('cardiology', 'cardiologist', 'primary care', 'pcp', 'nephrology', 'nephrologist', 'neurology',
                'neurologist', 'oncology', 'oncologist', 'surgery', 'surgeon', 'gastroenterology', 'hepatology',
                'pulmonology', 'endocrinology', 'urology', 'orthopedics', 'infectious disease', 'hematology',
                'rheumatology', 'psychiatry', 'dermatology', 'podiatry')
_ABSENT_TESTS = ('troponin', 'lactate', 'procalcitonin', 'ferritin', 'TSH', 'lipase', 'D-dimer',
                 'hemoglobin A1c', 'brain MRI', 'colonoscopy', 'echocardiogram', 'CT angiogram')
_NEG_SKIP = {'known', 'acute', 'other', 'evidence', 'significant', 'further', 'longer', 'change', 'changes',
             'history', 'prior', 'recent', 'more', 'need', 'signs', 'sign', 'complaints', 'issues', 'new'}
_FDA_DRUGS = ('furosemide', 'metoprolol', 'apixaban', 'lisinopril', 'atorvastatin', 'warfarin', 'metformin',
              'levetiracetam', 'pantoprazole', 'amlodipine')
_AMBIGUOUS = ('Medications?', 'Labs', 'Diagnosis?', 'Follow up', 'Vitals?', 'Allergies?')


def body(chunk):
    return _HEADER.sub('', chunk).strip()


_TRAILING = {'and', 'or', 'with', 'of', 'to', 'for', 'in', 'the', 'a', 'an', 'be', 'found', 'along',
             'after', 'both', 'one', 'at', 'on', 'by', 'from', 'that', 'which', 'who', 'was', 'is'}


def clean_phrase(text, max_words=8):
    """A short phrase ending on a whole clause: cut at the first comma or
    semicolon inside the word limit, then drop dangling function words."""
    text = re.sub(r'\s+', ' ', text).strip(' .,:;')
    if '___' in text or re.search(r'\d{3,}', text):
        return None
    words = text.split()[:max_words]
    for i, word in enumerate(words):
        if i >= 2 and word.endswith((',', ';')):
            words = words[:i + 1]
            break
    words = [w.strip(',;') for w in words]
    while words and words[-1].lower() in _TRAILING:
        words.pop()
    return ' '.join(words).lower() if len(words) >= 2 else None


def describe(sections):
    """(kind, phrase) naming the patient, from the note's own words."""
    if 'Chief Complaint' in sections:
        phrase = clean_phrase(re.split(r'[.\n]', body(sections['Chief Complaint'][0]))[0])
        if phrase:
            return 'presented with', phrase
    for name in ('History of Present Illness', 'Brief Hospital Course'):
        for chunk in sections.get(name, []):
            m = re.search(r'\b(?:presents|presented|presenting|admitted)\s+(?:with|for)\s+([^.;]{8,80})', chunk)
            if m and (phrase := clean_phrase(m.group(1))):
                return 'presented with', phrase
    if 'Discharge Diagnosis' in sections:
        text = re.sub(r'^(?:primary\s*(?:diagnosis(?:es)?)?(?:\s*at discharge)?\s*[:\-]?\s*)', '',
                      body(sections['Discharge Diagnosis'][0]), flags=re.IGNORECASE)
        text = re.sub(r'^\s*1[.)]\s*', '', text)
        phrase = clean_phrase(re.split(r'[.,;\n]| 2[.)] ', text)[0])
        if phrase:
            return 'discharged with a diagnosis of', phrase
    return None


def rel(note_id, *sections):
    return [{'note_id': note_id, 'section': s} for s in sections]


def candidates(note_id, sections, who):
    """Every draft this note supports, as (category, record without id)."""
    out = []
    subject = f'the patient who {who[0]} {who[1]}' if who[0] == 'presented with' else f'the patient {who[0]} {who[1]}'
    for chunk in sections.get('Pertinent Results', [])[:1]:
        for lab, pattern in _LABS.items():
            m = re.search(pattern, chunk)
            if m:
                out.append(('single_fact', {
                    'question': f'What was the first recorded {lab} of {subject}?',
                    'relevant': rel(note_id, 'Pertinent Results'), 'must_contain': [m.group(1)]}))
                break
    for chunk in sections.get('Physical Exam', [])[:1]:
        m = _BP.search(chunk)
        if m:
            out.append(('single_fact', {'question': f'What was the blood pressure on the admission exam of {subject}?',
                                            'relevant': rel(note_id, 'Physical Exam'), 'must_contain': [m.group(1)]}))
    meds = []
    for chunk in sections.get('Discharge Medications', [])[:1]:
        meds = list(dict.fromkeys(d.lower() for d in _MED.findall(chunk)))[:3]
        if len(meds) >= 2:
            out.append(('medication_list', {
                'question': f'Which medications was {subject} discharged on?',
                'relevant': rel(note_id, 'Discharge Medications'), 'must_contain': meds[:3]}))
    single = [rec for cat, rec in out if cat == 'single_fact' and rec['relevant'][0]['section'] == 'Pertinent Results']
    if single and len(meds) >= 1:
        lab_question = single[0]['question'].rstrip('?').replace('What was the first recorded ', '')
        out.append(('multi_section', {
            'question': f'What was the first recorded {lab_question}, and what was the first medication on the discharge list?',
            'relevant': rel(note_id, 'Pertinent Results', 'Discharge Medications'),
            'must_contain': single[0]['must_contain'] + meds[:1]}))
    negation = None
    for chunk in sections.get('History of Present Illness', []):
        for m in _NEG.finditer(chunk):
            finding = m.group(2).lower()
            if finding.split()[0] in _NEG_SKIP:
                continue
            negation = (f'{m.group(1).lower()} {finding}', finding)
            break
        if negation:
            break
    if negation:
        out.append(('negation', {
            'question': f'Did {subject} have {negation[1]}?',
            'relevant': rel(note_id, 'History of Present Illness'), 'must_contain': [negation[0]]}))
    note_text = ' '.join(c.lower() for chunks in sections.values() for c in chunks)
    start = sum(map(ord, note_id)) % len(_ABSENT_TESTS)   # rotate so the tests vary across notes
    for test in _ABSENT_TESTS[start:] + _ABSENT_TESTS[:start]:
        if test.lower() not in note_text and test.split()[-1].lower() not in note_text:
            out.append(('unanswerable', {'question': f'What was the {test} result for {subject}?', 'relevant': [],
                                             'must_contain': [], 'must_not_contain': []}))
            break
    for chunk in sections.get('Discharge Instructions', []):
        m = _FOLLOW.search(chunk)
        specialty = next((s for s in _SPECIALTIES if m and s in m.group(1).lower()), None)
        if specialty:
            out.append(('discharge_followup', {
                'question': f'Who should {subject} follow up with after discharge?',
                'relevant': rel(note_id, 'Discharge Instructions'), 'must_contain': [specialty]}))
            break
    return [(category, {**record, 'scope_note_id': note_id}) for category, record in out]


BEHAVIOR = {'unanswerable': 'refuse', 'fda_dosage': 'fda_lookup', 'ambiguous': 'clarify'}


def draft(corpus, targets, seed=20260928):
    by_note = defaultdict(lambda: defaultdict(list))
    for chunk, (note_id, section) in zip(corpus.chunks, corpus.keys):
        if note_id and section:
            by_note[note_id][section].append(chunk)
    named = {n: describe(s) for n, s in by_note.items()}
    counts = Counter(p for p in named.values() if p)
    unique = {n: p for n, p in named.items() if p and counts[p] == 1}

    pools = defaultdict(list)
    for note_id in sorted(unique):
        for category, record in candidates(note_id, by_note[note_id], unique[note_id]):
            pools[category].append(record)
    rng = random.Random(seed)
    chosen = []
    for category in CATEGORIES:
        want = targets.get(category, 0)
        if category == 'fda_dosage':
            picks = [{'question': q, 'relevant': [], 'must_contain': [], 'must_not_contain': []} for q in
                     (f'What is the usual dose of {d}?' for d in rng.sample(_FDA_DRUGS, min(want, len(_FDA_DRUGS))))]
        elif category == 'ambiguous':
            picks = [{'question': q, 'relevant': [], 'must_contain': [], 'must_not_contain': []}
                     for q in _AMBIGUOUS[:want]]
        else:
            pool = pools[category]
            picks = rng.sample(pool, min(want, len(pool)))
        for record in picks:
            chosen.append((category, record))
    rows = []
    for n, (category, record) in enumerate(chosen, 1):
        row = {'id': f'q{n:03d}', 'category': category, 'question': record['question'],
               'expected_behavior': BEHAVIOR.get(category, 'answer'), 'relevant': record['relevant'],
               'must_contain': record['must_contain'], 'must_not_contain': record.get('must_not_contain', []),
               'reviewed': False, 'notes': NOTE}
        if record.get('scope_note_id'):
            row['scope_note_id'] = record['scope_note_id']
        rows.append(row)
    stats = {'notes': len(by_note), 'named_uniquely': len(unique),
             'pool': {c: len(p) for c, p in pools.items()}, 'drafted': dict(Counter(c for c, _ in chosen))}
    return rows, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--corpus', choices=['demo', 'real'], default='real')
    parser.add_argument('--out', help='output path (default: the local real file, or a demo runs file)')
    parser.add_argument('--force', action='store_true', help='overwrite an existing output file')
    parser.add_argument('--show', type=int, default=0, help='print this many drafts (demo corpus only)')
    args = parser.parse_args(argv)
    if args.show and args.corpus == 'real':
        print('--show is only allowed for the demo corpus; real drafts stay in the local file.')
        return 2
    out = args.out or (GOLDEN_FILES['real'] if args.corpus == 'real'
                       else os.path.join('evals', 'runs', 'demo', 'drafted_demo_questions.jsonl'))
    if args.corpus == 'real' and '.local.' not in os.path.basename(out) and not out.startswith('outputs'):
        print('Real drafts may only be written to a *.local.* file or under outputs/.')
        return 2
    if os.path.exists(out) and not args.force:
        print(f'{out} exists; pass --force to overwrite it.')
        return 2
    rows, stats = draft(load_corpus(args.corpus), REAL_TARGETS)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    print(f"{stats['notes']} notes, {stats['named_uniquely']} with a unique patient description.")
    print('Candidates per category:', stats['pool'])
    print('Drafted per category:  ', stats['drafted'])
    print(f'Wrote {len(rows)} drafts (reviewed=false) to {out}.')
    for row in rows[:args.show]:
        print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
