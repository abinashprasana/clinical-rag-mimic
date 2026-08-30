import numpy as np
import pytest

import demo.runtime as demo_runtime
from demo.runtime import PUBLIC_DATASET_LABEL, run_turn


@pytest.fixture(autouse=False)
def force_offline(monkeypatch):
    """Forces the offline keyword-matching path regardless of whether a real
    GEMINI_API_KEY is present in the environment running the tests, so these
    tests are deterministic and don't depend on network access."""
    monkeypatch.setattr(demo_runtime, 'get_client', lambda: None)


def test_public_label_is_explicitly_fabricated():
    assert 'Synthetic' in PUBLIC_DATASET_LABEL
    assert 'no patient data' in PUBLIC_DATASET_LABEL


def test_diagnosis_query_is_grounded_in_fabricated_evidence(force_offline):
    result = run_turn('What discharge diagnoses are documented?')

    assert result['route'] == 'retrieve'
    assert result['reflection']['supported'] is True
    assert result['citations']
    assert 'fabricated demo record' in result['final_answer']
    assert all(citation['subject_id'] >= 9_000_000 for citation in result['citations'])


def test_public_runtime_refuses_dosing_advice_without_network_lookup():
    # The dosing-advice safety check runs before any client/Gemini lookup,
    # so this holds regardless of whether a key is configured.
    result = run_turn('What dose should I prescribe?')

    assert result['route'] == 'direct'
    assert result['citations'] == []
    assert 'does not provide prescribing or dosing advice' in result['final_answer']


def test_medication_question_does_not_collide_with_disposition(force_offline):
    # Regression test: "home" used to be a standalone trigger word for the
    # Discharge Disposition section, so a medications question containing
    # "at home" incorrectly matched disposition instead. See README's
    # Public deployment (Vercel) section for the full writeup.
    result = run_turn('What was the patient prescribed to take at home?')

    assert result['citations']
    top_section = result['citations'][0]['chunk_text'].split(':', 1)[0]
    assert top_section in ('Discharge Medications', 'Medications on Admission')


class _FakeEmbedding:
    def __init__(self, values):
        self.values = values


class _FakeEmbedResponse:
    def __init__(self, vectors):
        self.embeddings = [_FakeEmbedding(v) for v in vectors]


class _FakeGenerateResponse:
    def __init__(self, text):
        self.text = text


class _FakeClient:
    """Stands in for google.genai.Client so the Gemini-backed path can be
    tested without a real network call. embed_content returns a fixed
    all-ones vector regardless of input (cosine similarity then ranks purely
    by the header-boost term, which is enough to test the wiring)."""

    def __init__(self, generate_text):
        self._generate_text = generate_text
        self.models = self

    def embed_content(self, model, contents):
        dim = 3072
        return _FakeEmbedResponse([[1.0] * dim for _ in contents])

    def generate_content(self, model, contents, config):
        return _FakeGenerateResponse(self._generate_text)


def test_gemini_path_used_when_client_available(monkeypatch):
    fake_answer = (
        'The documented discharge diagnosis is community acquired pneumonia, '
        'right lower lobe, with type 2 diabetes mellitus and hyperglycemia.'
    )
    monkeypatch.setattr(demo_runtime, 'get_client', lambda: _FakeClient(fake_answer))

    result = run_turn('What is the discharge diagnosis?')

    assert 'Gemini' in result['tool_used']
    assert result['final_answer'] == fake_answer
    assert result['citations']


def test_falls_back_to_offline_path_when_gemini_call_raises(monkeypatch):
    class _BrokenClient:
        class models:
            @staticmethod
            def embed_content(*args, **kwargs):
                raise RuntimeError('simulated network failure')

    monkeypatch.setattr(demo_runtime, 'get_client', lambda: _BrokenClient())

    result = run_turn('What is the discharge diagnosis?')

    assert 'offline' in result['tool_used']
    assert result['citations']
