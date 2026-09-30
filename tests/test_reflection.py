"""Adversarial unit tests for agent/reflection.py's local faithfulness
check -- feeds it a fabricated claim not present in the retrieved chunks
and asserts it's caught, per the reflection research implemented here
(a dedicated grounding check should catch unsupported claims rather than
silently returning them)."""
from agent.reflection import local_reflect

CHUNKS = [
    {'chunk_text': 'Patient was discharged home in stable condition with a '
                    'diagnosis of community acquired pneumonia. Discharge '
                    'medications include azithromycin 250 mg daily.'}
]


def test_supported_claim_passes():
    answer = 'The discharge diagnosis was community acquired pneumonia.'
    result = local_reflect(answer, CHUNKS)
    assert result['supported'] is True
    assert result['unsupported_claims'] == []


def test_fabricated_numeric_claim_is_caught():
    # 500 mg never appears in the retrieved chunk (only 250 mg does) --
    # a fabricated dose is exactly the highest-stakes failure mode.
    answer = 'The patient was prescribed azithromycin 500 mg daily.'
    result = local_reflect(answer, CHUNKS)
    assert result['supported'] is False
    assert result['unsupported_claims']


def test_fabricated_unrelated_claim_is_caught():
    answer = 'The patient underwent emergency cardiac bypass surgery.'
    result = local_reflect(answer, CHUNKS)
    assert result['supported'] is False


def test_explicit_refusal_is_always_supported():
    answer = 'I cannot find this information in the provided notes.'
    result = local_reflect(answer, CHUNKS)
    assert result['supported'] is True


def test_no_retrieved_chunks_skips_check():
    result = local_reflect('Some direct-response answer with no retrieval.', [])
    assert result['supported'] is True


def test_empty_answer_is_supported_trivially():
    result = local_reflect('', CHUNKS)
    assert result['supported'] is True


# Fabricated passages for the rules added after the stress test.
NOTE = [
    '[Discharge Medications] 1. Donepezil 10 mg PO daily 2. Cephalexin 500 mg PO QID',
    '[Pertinent Results] Sodium 136, creatinine 0.9, no ketones present.',
    '[Physical Exam] Vitals: BP 138/84, HR 82, RR 26. No chest pain, no fevers.',
    '[History of Present Illness] An 84 year old with a urinary tract infection.',
]


def test_number_from_elsewhere_in_the_note_is_caught():
    assert local_reflect('Donepezil 500 mg PO daily.', NOTE)['supported'] is False
    assert local_reflect('Sodium was 0.9.', NOTE)['supported'] is False
    assert local_reflect('Donepezil 10 mg PO daily.', NOTE)['supported'] is True


def test_list_numbering_is_not_a_value():
    # "1." sits next to Donepezil in the note, but only as list numbering
    assert local_reflect('Donepezil 1 mg PO daily.', NOTE)['supported'] is False
    quoted = '1. Donepezil 10 mg PO daily 2. Cephalexin 500 mg PO QID'
    assert local_reflect(quoted, NOTE)['supported'] is True


def test_finding_recorded_as_absent_cannot_be_asserted():
    assert local_reflect('The patient had chest pain and fevers.', NOTE)['supported'] is False
    assert local_reflect('No chest pain and no fevers were recorded.', NOTE)['supported'] is True


def test_paraphrased_vitals_and_ages_pass():
    assert local_reflect('Vitals on exam: respiratory rate 26.', NOTE)['supported'] is True
    assert local_reflect('Vitals on exam: blood pressure 138/84.', NOTE)['supported'] is True
    assert local_reflect('Vitals on exam: blood pressure 136/84.', NOTE)['supported'] is False
    assert local_reflect('The 84 year old had a urinary tract infection.', NOTE)['supported'] is True


def test_lone_value_is_still_checked():
    assert local_reflect('31.', NOTE)['supported'] is False
    assert local_reflect('8.4.', NOTE)['supported'] is False
    assert local_reflect('26.', NOTE)['supported'] is True   # RR 26, not list numbering
    assert local_reflect('136.', NOTE)['supported'] is True
