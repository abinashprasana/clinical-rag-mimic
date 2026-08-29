"""Unit tests for agent/tools.py. Mocks the openFDA HTTP call so the suite
is deterministic and doesn't depend on network access or the live API."""
from unittest.mock import patch, Mock
from agent.tools import fda_label_tool


def _fake_response(results):
    resp = Mock()
    resp.status_code = 200
    resp.json.return_value = {'results': results}
    return resp


def _label_record(generic='METFORMIN', brand='Metformin'):
    return {
        'openfda': {'brand_name': [brand], 'generic_name': [generic]},
        'dosage_and_administration': ['Take as directed.'],
        'contraindications': ['Severe renal impairment.'],
        'warnings': ['Lactic acidosis risk.'],
    }


def test_fda_label_tool_exact_match():
    with patch('agent.tools.requests.get', return_value=_fake_response([_label_record()])):
        result = fda_label_tool('metformin')
    assert result is not None
    assert result['matched_generic_name'] == 'METFORMIN'
    assert result['dosage_and_administration'] == 'Take as directed.'
    assert result['contraindications'] == 'Severe renal impairment.'
    assert result['source'] == 'openFDA Drug Label API'


def test_fda_label_tool_not_found_returns_none():
    with patch('agent.tools.requests.get', return_value=_fake_response([])):
        result = fda_label_tool('zzzznotarealdrugxyz')
    assert result is None


def test_fda_label_tool_empty_input_returns_none_without_request():
    with patch('agent.tools.requests.get') as mock_get:
        result = fda_label_tool('   ')
    assert result is None
    mock_get.assert_not_called()


def test_fda_label_tool_network_failure_returns_none():
    import requests
    with patch('agent.tools.requests.get', side_effect=requests.RequestException('boom')):
        result = fda_label_tool('metformin')
    assert result is None


def test_fda_label_tool_falls_back_to_non_exact_match():
    # First three calls (exact brand/generic/substance) return nothing;
    # the fourth (non-exact brand) finally matches.
    responses = [_fake_response([]), _fake_response([]), _fake_response([]), _fake_response([_label_record()])]
    with patch('agent.tools.requests.get', side_effect=responses):
        result = fda_label_tool('metformin')
    assert result is not None
    assert result['matched_generic_name'] == 'METFORMIN'
