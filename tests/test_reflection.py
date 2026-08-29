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
