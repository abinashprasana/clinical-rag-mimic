"""Golden set loader and validator. Uses only the fabricated demo corpus and
hand-made records; no real note text."""
import json

from evals.corpus import Corpus, canonical_section, chunk_section, load_corpus
from evals.golden import (
    DEMO_TARGET_COUNTS,
    GOLDEN_FILES,
    contains,
    load_questions,
    normalise,
    select,
    validate,
)

CHUNKS = [
    '[Discharge Medications] Furosemide 40 mg PO daily. Metoprolol 25 mg PO BID.',
    '[History of Present Illness] Patient denies chest pain.',
    'no header here, fixed-size fallback chunk',
]
PROVENANCE = [{'subject_id': 1, 'hadm_id': 100}, {'subject_id': 1, 'hadm_id': 100},
              {'subject_id': 2, 'hadm_id': 200}]
TINY = Corpus('tiny', CHUNKS, PROVENANCE, output_dir=None)


def record(**overrides):
    base = {
        'id': 'q001', 'category': 'medication_list', 'question': 'Which drugs?',
        'expected_behavior': 'answer',
        'relevant': [{'note_id': '100', 'section': 'Discharge Medications'}],
        'must_contain': ['furosemide 40 mg'], 'must_not_contain': [],
        'reviewed': False, 'notes': '',
    }
    base.update(overrides)
    return base


def test_section_canonicalisation():
    assert canonical_section('DISCHARGE MEDICATIONS') == 'Discharge Medications'
    assert canonical_section('Follow-up Instructions') == 'Followup Instructions'
    assert canonical_section('Not A Header') is None
    assert chunk_section(CHUNKS[2]) is None
    assert TINY.keys[0] == ('100', 'Discharge Medications')


def test_normalise_keeps_numbers_and_units_exact():
    assert normalise('  Furosemide\n40  MG ') == 'furosemide 40 mg'
    assert contains('Take Furosemide 40 mg daily', 'furosemide 40 mg')
    assert not contains('Take Furosemide 40mg daily', 'furosemide 40 mg')
    assert not contains('Furosemide 140 mg', 'furosemide 40 mg')
    assert contains('non–adherence', 'non-adherence')


def test_valid_record_passes():
    errors, warnings = validate([record()], TINY)
    assert errors == [] and warnings == []


def test_schema_errors_are_reported():
    bad = record(id='x1', expected_behavior='maybe', extra=1)
    del bad['notes']
    messages = [m for _, m in validate([bad], TINY)[0]]
    assert any('missing field notes' in m for m in messages)
    assert any('unknown field extra' in m for m in messages)
    assert any('id must look like' in m for m in messages)
    assert any('expected_behavior must be' in m for m in messages)


def test_relevant_rules():
    missing = record(relevant=[{'note_id': '100', 'section': 'Allergies'}])
    assert any('no indexed Allergies' in m for _, m in validate([missing], TINY)[0])
    refuse_with_gold = record(category='unanswerable', expected_behavior='refuse')
    assert any('must have an empty relevant' in m for _, m in validate([refuse_with_gold], TINY)[0])
    answer_without_gold = record(relevant=[])
    assert any('at least one relevant' in m for _, m in validate([answer_without_gold], TINY)[0])
    wrong_behaviour = record(category='ambiguous', expected_behavior='answer')
    assert any('expects behaviour clarify' in m for _, m in validate([wrong_behaviour], TINY)[0])


def test_duplicate_ids_and_gold_term_warning():
    errors, _ = validate([record(), record()], TINY)
    assert ('q001', 'duplicate id') in errors
    _, warnings = validate([record(must_contain=['furosemide 80 mg'])], TINY)
    assert warnings == [('q001', 'must_contain[0] not found in the gold passages')]


def test_select_reviewed_only_by_default():
    questions = [record(id='q001', reviewed=True), record(id='q002')]
    assert [q['id'] for q in select(questions)] == ['q001']
    assert len(select(questions, include_unreviewed=True)) == 2


def test_demo_golden_set_is_valid_against_demo_index():
    questions = load_questions(GOLDEN_FILES['demo'])
    errors, warnings = validate(questions, load_corpus('demo'))
    assert errors == [] and warnings == []
    counts = {c: sum(q['category'] == c for q in questions) for c in DEMO_TARGET_COUNTS}
    assert counts == DEMO_TARGET_COUNTS


def test_template_has_placeholder_text_only():
    with open('evals/golden/real_questions.template.jsonl', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1
    assert rows[0]['question'] == 'WRITE YOUR QUESTION HERE'
    assert rows[0]['reviewed'] is False
