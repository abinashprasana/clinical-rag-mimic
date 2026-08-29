import logging
import uuid

import pytest

import app as app_module


@pytest.fixture
def flask_app(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.config, 'OUTPUT_DIR', str(tmp_path))
    monkeypatch.setattr(app_module, 'SentenceTransformer', lambda _name: object())
    monkeypatch.setattr(app_module, 'load_index', lambda _path: (object(), [], []))
    monkeypatch.setattr(app_module, 'load_generator', lambda: object())
    monkeypatch.setattr(app_module, 'get_graph', lambda *_args: object())
    monkeypatch.setattr(app_module, 'run_turn', lambda *_args, **_kwargs: _result())

    application = app_module.create_app()
    application.config.update(
        TESTING=True,
        SECRET_KEY='test-secret',
        PROPAGATE_EXCEPTIONS=False,
    )
    return application


def _result(**overrides):
    result = {
        'final_answer': 'Grounded answer.',
        'route': 'retrieve',
        'tool_used': 'Retrieved 1 evidence passages',
        'citations': [{'chunk_text': 'Evidence.', 'score': 0.91}],
        'reflection': {'supported': True, 'unsupported_claims': []},
        'needs_clarification': False,
        'fda_result': None,
    }
    result.update(overrides)
    return result


def test_get_index_passes_non_sensitive_metadata_and_security_headers(
        flask_app, monkeypatch):
    captured = {}

    def fake_render(template_name, **context):
        captured['template_name'] = template_name
        captured['context'] = context
        return 'index'

    monkeypatch.setattr(app_module, 'render_template', fake_render)

    response = flask_app.test_client().get('/')

    assert response.status_code == 200
    assert response.get_data(as_text=True) == 'index'
    assert captured == {
        'template_name': 'index.html',
        'context': {
            'accuracy_pct': None,
            'accuracy_label': None,
            'dataset_version': 'MIMIC-IV-Note v2.2',
            'retrieval_top_k': app_module.config.DEFAULT_TOP_K,
            'generator_model': app_module.config.LOCAL_GENERATOR_MODEL,
            'public_demo': False,
        },
    }
    assert response.headers['Content-Security-Policy'] == (
        "default-src 'self'; base-uri 'self'; object-src 'none'; "
        "frame-ancestors 'none'; form-action 'self'; connect-src 'self'; "
        "img-src 'self'; font-src 'self'; script-src 'self'; style-src 'self'"
    )
    assert response.headers['X-Content-Type-Options'] == 'nosniff'
    assert response.headers['X-Frame-Options'] == 'DENY'
    assert response.headers['Referrer-Policy'] == 'no-referrer'
    assert response.headers['Cache-Control'] == 'no-store'


@pytest.mark.parametrize(
    'result',
    [
        _result(),
        _result(
            final_answer='Which admission should I review?',
            route='clarify',
            tool_used='Clarification requested',
            citations=[],
            reflection=None,
            needs_clarification=True,
        ),
        _result(
            final_answer=None,
            route='dosage',
            tool_used='FDA Label Lookup: metformin',
            citations=[],
            reflection=None,
            fda_result={
                'drug_name': 'metformin',
                'dosage_and_administration': 'Use as directed.',
            },
        ),
    ],
    ids=['retrieval', 'clarification', 'fda'],
)
def test_ask_preserves_success_contract_across_branches(
        flask_app, monkeypatch, result):
    monkeypatch.setattr(app_module, 'run_turn', lambda *_args, **_kwargs: result)

    response = flask_app.test_client().post('/ask', json={'question': 'Question?'})

    assert response.status_code == 200
    assert response.get_json() == {
        'answer': result['final_answer'],
        'route': result['route'],
        'tool_used': result['tool_used'],
        'citations': result['citations'],
        'reflection': result['reflection'],
        'needs_clarification': result['needs_clarification'],
        'fda_result': result['fda_result'],
    }


@pytest.mark.parametrize('question', [None, '', '   ', 42, []])
def test_ask_rejects_empty_or_non_string_questions(
        flask_app, monkeypatch, question):
    called = False

    def fake_run_turn(*_args, **_kwargs):
        nonlocal called
        called = True
        return _result()

    monkeypatch.setattr(app_module, 'run_turn', fake_run_turn)

    response = flask_app.test_client().post('/ask', json={'question': question})

    assert response.status_code == 400
    assert response.get_json() == {
        'error': {
            'code': 'INVALID_QUESTION',
            'message': app_module.ERROR_MESSAGES['INVALID_QUESTION'],
        }
    }
    assert called is False


@pytest.mark.parametrize(
    'request_kwargs',
    [
        {},
        {'data': '{broken', 'content_type': 'application/json'},
        {'json': []},
        {'json': None},
    ],
    ids=['missing', 'malformed', 'array', 'null'],
)
def test_ask_rejects_missing_or_malformed_json(flask_app, request_kwargs):
    response = flask_app.test_client().post('/ask', **request_kwargs)

    assert response.status_code == 400
    assert response.get_json()['error']['code'] == 'REQUEST_FAILED'


@pytest.mark.parametrize('thread_id', [None, '', 'not-a-uuid', 123, []])
def test_ask_rejects_invalid_client_thread_ids(
        flask_app, monkeypatch, thread_id):
    called = False

    def fake_run_turn(*_args, **_kwargs):
        nonlocal called
        called = True
        return _result()

    monkeypatch.setattr(app_module, 'run_turn', fake_run_turn)

    response = flask_app.test_client().post(
        '/ask', json={'question': 'Question?', 'thread_id': thread_id})

    assert response.status_code == 400
    assert response.get_json()['error']['code'] == 'REQUEST_FAILED'
    assert called is False


def test_client_thread_ids_are_namespaced_and_isolate_tabs(
        flask_app, monkeypatch):
    thread_ids = []

    def fake_run_turn(_question, thread_id):
        thread_ids.append(thread_id)
        return _result()

    monkeypatch.setattr(app_module, 'run_turn', fake_run_turn)
    client = flask_app.test_client()
    first_tab = str(uuid.uuid4())
    second_tab = str(uuid.uuid4())

    for tab_id in (first_tab, second_tab, first_tab):
        response = client.post(
            '/ask', json={'question': 'Question?', 'thread_id': tab_id})
        assert response.status_code == 200

    with client.session_transaction() as browser_session:
        session_id = browser_session['session_id']

    assert thread_ids == [
        f'{session_id}:{first_tab}',
        f'{session_id}:{second_tab}',
        f'{session_id}:{first_tab}',
    ]
    assert thread_ids[0] != thread_ids[1]


def test_same_client_thread_id_is_isolated_between_browser_sessions(
        flask_app, monkeypatch):
    thread_ids = []

    def fake_run_turn(_question, thread_id):
        thread_ids.append(thread_id)
        return _result()

    monkeypatch.setattr(app_module, 'run_turn', fake_run_turn)
    tab_id = str(uuid.uuid4())

    for client in (flask_app.test_client(), flask_app.test_client()):
        response = client.post(
            '/ask', json={'question': 'Question?', 'thread_id': tab_id})
        assert response.status_code == 200

    assert thread_ids[0].endswith(f':{tab_id}')
    assert thread_ids[1].endswith(f':{tab_id}')
    assert thread_ids[0] != thread_ids[1]


def test_legacy_ask_uses_session_only_thread_id(flask_app, monkeypatch):
    observed = {}

    def fake_run_turn(question, thread_id):
        observed.update(question=question, thread_id=thread_id)
        return _result()

    monkeypatch.setattr(app_module, 'run_turn', fake_run_turn)
    client = flask_app.test_client()

    response = client.post('/ask', json={'question': '  Question?  '})

    with client.session_transaction() as browser_session:
        session_id = browser_session['session_id']
    assert response.status_code == 200
    assert observed == {'question': 'Question?', 'thread_id': session_id}


def test_operational_failure_is_logged_and_sanitized(
        flask_app, monkeypatch, caplog):
    secret = 'private exception detail'

    def fail_run_turn(*_args, **_kwargs):
        raise RuntimeError(secret)

    monkeypatch.setattr(app_module, 'run_turn', fail_run_turn)

    with caplog.at_level(logging.ERROR):
        response = flask_app.test_client().post(
            '/ask', json={'question': 'Question?'})

    assert response.status_code == 500
    assert response.get_json() == {
        'error': {
            'code': 'REQUEST_FAILED',
            'message': app_module.ERROR_MESSAGES['REQUEST_FAILED'],
        }
    }
    assert secret not in response.get_data(as_text=True)
    assert 'Clinical request failed' in caplog.text


def test_unexpected_failure_uses_internal_error_fallback(
        flask_app, monkeypatch, caplog):
    secret = 'private framework detail'

    def fail_render(*_args, **_kwargs):
        raise RuntimeError(secret)

    monkeypatch.setattr(app_module, 'render_template', fail_render)

    with caplog.at_level(logging.ERROR):
        response = flask_app.test_client().get('/')

    assert response.status_code == 500
    assert response.get_json() == {
        'error': {
            'code': 'INTERNAL_ERROR',
            'message': app_module.ERROR_MESSAGES['INTERNAL_ERROR'],
        }
    }
    assert secret not in response.get_data(as_text=True)
    assert 'Unhandled application error' in caplog.text


def test_generate_node_reports_retrieved_evidence_passages(monkeypatch):
    graph_module = pytest.importorskip('agent.graph')
    monkeypatch.setattr(
        graph_module,
        'generate_answer',
        lambda _question, _chunks, _generator: ('Grounded answer.', 0.1),
    )
    state = {
        'question': 'Question?',
        'step_count': 0,
        'fda_result': None,
        'retrieved_chunks': [
            {'chunk_text': 'First passage.'},
            {'chunk_text': 'Second passage.'},
        ],
    }

    result = graph_module.make_generate_node(object())(state)

    assert result['tool_used'] == 'Retrieved 2 evidence passages'


def test_module_level_vercel_app_is_fabricated_and_healthy():
    application = app_module.app
    assert application.config['PUBLIC_DEMO'] is True

    response = application.test_client().get('/healthz')

    assert response.status_code == 200
    body = response.get_json()
    assert body['status'] == 'ok'
    assert body['runtime'] == 'synthetic-demo'
    assert body['data_boundary'] == 'fabricated-only'
    # The public runtime now retrieves and generates via the Gemini API (see
    # demo_runtime.py), falling back to an offline method only if no key is
    # configured or a call fails -- so this reflects whether a key is set,
    # not a hardcoded False.
    assert body['external_ai'] == bool(app_module.config.GEMINI_API_KEY)


def test_public_demo_question_returns_only_synthetic_evidence():
    response = app_module.app.test_client().post(
        '/ask',
        json={'question': 'What discharge diagnoses are documented?'},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['route'] == 'retrieve'
    assert payload['reflection']['supported'] is True
    assert payload['citations']
    assert all(item['subject_id'] >= 9_000_000 for item in payload['citations'])
    assert all(item['hadm_id'] >= 29_000_000 for item in payload['citations'])
