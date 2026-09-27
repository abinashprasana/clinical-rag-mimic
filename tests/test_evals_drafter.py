"""Local question drafter, on fabricated notes only."""
import json

from evals.corpus import Corpus
from evals.draft_real_questions import clean_phrase, draft, main

CHUNKS = [
    '[Chief Complaint] Worsening shortness of breath and leg swelling for one week, seen in the emergency department today.',
    '[History of Present Illness] Patient presented with leg swelling. She denies chest pain and reports orthopnea at night.',
    '[Pertinent Results] 06:10AM BLOOD WBC-9.4 Hgb-11.2 Creat-1.6 Na-134 K-4.2',
    '[Physical Exam] VS: T 98.2 BP 148/90 HR 102 RR 20, lungs with bibasilar crackles and pitting edema.',
    '[Discharge Medications] 1. Furosemide 60 mg PO daily 2. Metoprolol 50 mg PO daily 3. Apixaban 5 mg PO BID',
    '[Discharge Instructions] Please follow up with your cardiologist in two weeks and weigh yourself every day.',
]
PROVENANCE = [{'subject_id': 1, 'hadm_id': 111}] * len(CHUNKS)
CORPUS = Corpus('tiny', CHUNKS, PROVENANCE, output_dir=None)


def test_clean_phrase_ends_on_a_whole_clause():
    assert clean_phrase('two days of black, tarry stools and one episode') == 'two days of black'
    assert clean_phrase('redness and swelling of the left lower leg after') == 'redness and swelling of the left lower leg'
    assert clean_phrase('pain in ___ region') is None


def test_drafts_come_from_the_note_and_start_unreviewed():
    targets = {'single_fact': 2, 'medication_list': 1, 'multi_section': 1, 'negation': 1,
               'unanswerable': 1, 'discharge_followup': 1, 'fda_dosage': 1, 'ambiguous': 1}
    rows, stats = draft(CORPUS, targets)
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r['category'], []).append(r)
        assert r['reviewed'] is False
    assert stats['named_uniquely'] == 1
    assert by_cat['medication_list'][0]['must_contain'] == ['furosemide', 'metoprolol', 'apixaban']
    assert by_cat['negation'][0]['must_contain'] == ['denies chest pain']
    assert by_cat['discharge_followup'][0]['must_contain'] == ['cardiologist']
    assert by_cat['unanswerable'][0]['expected_behavior'] == 'refuse'
    assert {r['must_contain'][0] for r in by_cat['single_fact']} <= {'9.4', '148/90'}


def test_real_corpus_never_prints_and_never_writes_a_tracked_file(tmp_path):
    assert main(['--corpus', 'real', '--show', '3']) == 2
    assert main(['--corpus', 'real', '--out', str(tmp_path / 'questions.jsonl')]) == 2


def test_demo_run_writes_valid_json_lines(tmp_path):
    out = tmp_path / 'drafts.jsonl'
    assert main(['--corpus', 'demo', '--out', str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines()]
    assert rows and all(r['reviewed'] is False for r in rows)
