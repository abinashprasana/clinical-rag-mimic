import json
import os
import secrets
import uuid

from flask import Flask, jsonify, render_template, request, session
from werkzeug.exceptions import HTTPException

import config


PUBLIC_RUNTIME = 'vercel-demo'
LOCAL_RUNTIME = 'local'

# Full-pipeline dependencies are loaded only for the local runtime. Keeping
# them out of module import makes the public Vercel function small and avoids
# downloading model weights during a cold start. These names remain module
# attributes so the local unit tests can replace them with lightweight fakes.
SentenceTransformer = None
load_index = None
load_generator = None
get_graph = None
run_turn = None


ERROR_MESSAGES = {
    'INVALID_QUESTION': 'Please enter a valid clinical question.',
    'REQUEST_FAILED': 'We could not complete this request. Please try again.',
    'INTERNAL_ERROR': 'An unexpected error occurred. Please try again.',
}


def _error_response(code, status):
    return jsonify({'error': {'code': code, 'message': ERROR_MESSAGES[code]}}), status


def _load_local_dependencies():
    """Import the heavyweight local stack only when it is actually requested."""
    global SentenceTransformer, load_index, load_generator, get_graph, run_turn

    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as transformer
        SentenceTransformer = transformer
    if load_index is None:
        from retrieval import load_index as index_loader
        load_index = index_loader
    if load_generator is None:
        from generation import load_generator as generator_loader
        load_generator = generator_loader
    if get_graph is None or run_turn is None:
        from agent.graph import get_graph as graph_loader, run_turn as graph_runner
        if get_graph is None:
            get_graph = graph_loader
        if run_turn is None:
            run_turn = graph_runner


def _initialize_runtime(runtime):
    if runtime == PUBLIC_RUNTIME:
        from demo_runtime import (
            PUBLIC_ACCURACY_LABEL,
            PUBLIC_ACCURACY_PCT,
            PUBLIC_DATASET_LABEL,
            PUBLIC_GENERATOR_LABEL,
            run_turn as demo_runner,
        )
        return {
            'runner': demo_runner,
            'dataset_label': PUBLIC_DATASET_LABEL,
            'generator_label': PUBLIC_GENERATOR_LABEL,
            'output_dir': None,
            'accuracy_pct': PUBLIC_ACCURACY_PCT,
            'accuracy_label': PUBLIC_ACCURACY_LABEL,
            'public_demo': True,
        }

    if runtime != LOCAL_RUNTIME:
        raise ValueError(f'Unsupported application runtime: {runtime}')

    _load_local_dependencies()
    print('Initializing local models and loading local data...')
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    index_loaded, chunks_loaded, provenance_loaded = load_index(config.OUTPUT_DIR)
    generator = load_generator()
    get_graph(model, index_loaded, chunks_loaded, provenance_loaded, generator)
    print('Local application ready to serve requests.')
    return {
        # Resolve the module global at request time so tests and local tooling
        # can replace the graph runner after the Flask app has been created.
        'runner': lambda question, thread_id: run_turn(question, thread_id=thread_id),
        'dataset_label': config.DATASET_LABEL,
        'generator_label': config.LOCAL_GENERATOR_MODEL,
        'output_dir': config.OUTPUT_DIR,
        'accuracy_pct': None,
        'accuracy_label': None,
        'public_demo': False,
    }


def create_app(runtime=LOCAL_RUNTIME):
    runtime_state = _initialize_runtime(runtime)
    app = Flask(__name__)
    app.secret_key = (
        secrets.token_hex(32)
        if runtime_state['public_demo']
        else config.FLASK_SECRET_KEY
    )
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax',
        PUBLIC_DEMO=runtime_state['public_demo'],
    )

    @app.after_request
    def add_security_headers(response):
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "base-uri 'self'; "
            "object-src 'none'; "
            "frame-ancestors 'none'; "
            "form-action 'self'; "
            "connect-src 'self'; "
            "img-src 'self'; "
            "font-src 'self'; "
            "script-src 'self'; "
            "style-src 'self'"
        )
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Permissions-Policy'] = (
            'camera=(), microphone=(), geolocation=(), payment=(), usb=()'
        )
        response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
        response.headers['Cross-Origin-Resource-Policy'] = 'same-origin'
        if request.endpoint != 'static':
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        if isinstance(error, HTTPException):
            return error
        app.logger.exception('Unhandled application error')
        return _error_response('INTERNAL_ERROR', 500)

    @app.route('/')
    def index():
        accuracy_pct = runtime_state['accuracy_pct']
        output_dir = runtime_state['output_dir']
        if accuracy_pct is None and output_dir:
            eval_path = os.path.join(output_dir, 'evaluation_results.json')
            if os.path.exists(eval_path):
                with open(eval_path, encoding='utf-8') as handle:
                    results = json.load(handle)
                if results:
                    hits = sum(item['keyword_found'] for item in results)
                    accuracy_pct = round(hits / len(results) * 100)
        return render_template(
            'index.html',
            accuracy_pct=accuracy_pct,
            accuracy_label=runtime_state['accuracy_label'],
            dataset_version=runtime_state['dataset_label'],
            retrieval_top_k=config.DEFAULT_TOP_K,
            generator_model=runtime_state['generator_label'],
            public_demo=runtime_state['public_demo'],
            gemini_enabled=bool(config.GEMINI_API_KEY),
        )

    @app.route('/healthz')
    def healthz():
        return jsonify({
            'status': 'ok',
            'runtime': 'synthetic-demo' if runtime_state['public_demo'] else 'local-full',
            'data_boundary': (
                'fabricated-only' if runtime_state['public_demo'] else 'local-configured'
            ),
            'external_ai': bool(config.GEMINI_API_KEY) if runtime_state['public_demo'] else bool(
                config.ENABLE_EXTERNAL_REFLECTION and config.GEMINI_API_KEY
            ),
        })

    @app.route('/ask', methods=['POST'])
    def ask():
        if not request.is_json:
            return _error_response('REQUEST_FAILED', 400)

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error_response('REQUEST_FAILED', 400)

        question = data.get('question')
        if not isinstance(question, str) or not question.strip():
            return _error_response('INVALID_QUESTION', 400)
        question = question.strip()
        if len(question) > 4000:
            return _error_response('INVALID_QUESTION', 400)

        client_thread_id = data.get('thread_id') if 'thread_id' in data else None
        if 'thread_id' in data:
            if not isinstance(client_thread_id, str):
                return _error_response('REQUEST_FAILED', 400)
            try:
                client_thread_id = str(uuid.UUID(client_thread_id))
            except (ValueError, AttributeError):
                return _error_response('REQUEST_FAILED', 400)

        if runtime_state['public_demo']:
            # Stateless by design: any Vercel function instance can handle the
            # next request without depending on process-local conversation data.
            thread_id = client_thread_id or str(uuid.uuid4())
        else:
            if 'session_id' not in session:
                session['session_id'] = str(uuid.uuid4())
            thread_id = session['session_id']
            if client_thread_id is not None:
                thread_id = f'{thread_id}:{client_thread_id}'

        try:
            result = runtime_state['runner'](question, thread_id=thread_id)
            return jsonify({
                'answer': result.get('final_answer'),
                'route': result.get('route'),
                'tool_used': result.get('tool_used'),
                'citations': result.get('citations', []),
                'reflection': result.get('reflection'),
                'needs_clarification': result.get('needs_clarification', False),
                'fda_result': result.get('fda_result'),
            })
        except Exception:
            app.logger.exception('Clinical request failed')
            return _error_response('REQUEST_FAILED', 500)

    return app


# Vercel's Flask detector requires a module-level WSGI instance named `app`.
# The exported application is deliberately pinned to the fabricated, stateless
# demo runtime and therefore cannot be switched to restricted data by an
# environment-variable mistake.
app = create_app(runtime=PUBLIC_RUNTIME)


if __name__ == '__main__':
    selected_runtime = os.getenv('APP_RUNTIME', LOCAL_RUNTIME).strip().lower()
    local_app = create_app(runtime=selected_runtime)
    if config.FLASK_DEBUG:
        local_app.run(
            host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            debug=True,
            use_reloader=False,
        )
    else:
        # waitress is kept for the full local pipeline. The public deployment
        # uses Vercel's WSGI runtime and never imports waitress.
        from waitress import serve
        print(f'Serving on http://{config.FLASK_HOST}:{config.FLASK_PORT} (waitress)')
        serve(local_app, host=config.FLASK_HOST, port=config.FLASK_PORT, threads=4)
