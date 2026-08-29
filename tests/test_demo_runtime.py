from demo_runtime import PUBLIC_DATASET_LABEL, run_turn


def test_public_label_is_explicitly_fabricated():
    assert 'Synthetic' in PUBLIC_DATASET_LABEL
    assert 'no patient data' in PUBLIC_DATASET_LABEL


def test_diagnosis_query_is_grounded_in_fabricated_evidence():
    result = run_turn('What discharge diagnoses are documented?')

    assert result['route'] == 'retrieve'
    assert result['reflection']['supported'] is True
    assert result['citations']
    assert 'fabricated demo record' in result['final_answer']
    assert all(citation['subject_id'] >= 9_000_000 for citation in result['citations'])


def test_public_runtime_refuses_dosing_advice_without_network_lookup():
    result = run_turn('What dose should I prescribe?')

    assert result['route'] == 'direct'
    assert result['citations'] == []
    assert 'does not provide prescribing or dosing advice' in result['final_answer']


def test_medication_question_does_not_collide_with_disposition():
    # Regression test: "home" used to be a standalone trigger word for the
    # Discharge Disposition section, so a medications question containing
    # "at home" incorrectly matched disposition instead. See README's
    # Public deployment (Vercel) section for the full writeup.
    result = run_turn('What was the patient prescribed to take at home?')

    assert result['citations']
    top_section = result['citations'][0]['chunk_text'].split(':', 1)[0]
    assert top_section in ('Discharge Medications', 'Medications on Admission')
