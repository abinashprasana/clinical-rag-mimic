import json
import os
import uuid
from flask import Flask, render_template, request, jsonify, session
from werkzeug.exceptions import HTTPException
from sentence_transformers import SentenceTransformer
from retrieval import load_index
from generation import load_generator
from agent.graph import get_graph, run_turn
import config


ERROR_MESSAGES = {
    'INVALID_QUESTION': 'Please enter a valid clinical question.',
    'REQUEST_FAILED': 'We could not complete this request. Please try again.',
    'INTERNAL_ERROR': 'An unexpected error occurred. Please try again.',
}


def _error_response(code, status):
    return jsonify({'error': {'code': code, 'message': ERROR_MESSAGES[code]}}), status


def create_app():
    app = Flask(__name__)
    app.secret_key = config.FLASK_SECRET_KEY
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax')

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

    print("Initializing models and loading data...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    index_loaded, chunks_loaded, provenance_loaded = load_index(config.OUTPUT_DIR)
    generator = load_generator()
    get_graph(model, index_loaded, chunks_loaded, provenance_loaded, generator)
    print("App ready to serve requests.")

    @app.route('/')
    def index():
        # Single source of truth for the accuracy figure shown in the UI --
        # computed live from the eval results file rather than hardcoded
        # copy, so it can never drift out of sync with the actual numbers.
        accuracy_pct = None
        eval_path = os.path.join(config.OUTPUT_DIR, 'evaluation_results.json')
        if os.path.exists(eval_path):
            with open(eval_path, encoding='utf-8') as f:
                results = json.load(f)
            if results:
                hits = sum(r['keyword_found'] for r in results)
                accuracy_pct = round(hits / len(results) * 100)
        return render_template(
            'index.html',
            accuracy_pct=accuracy_pct,
            dataset_version=config.DATASET_LABEL,
            retrieval_top_k=config.DEFAULT_TOP_K,
            generator_model=config.LOCAL_GENERATOR_MODEL,
        )

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

        client_thread_id = data.get('thread_id') if 'thread_id' in data else None
        if 'thread_id' in data:
            if not isinstance(client_thread_id, str):
                return _error_response('REQUEST_FAILED', 400)
            try:
                client_thread_id = str(uuid.UUID(client_thread_id))
            except (ValueError, AttributeError):
                return _error_response('REQUEST_FAILED', 400)

        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())

        thread_id = session['session_id']
        if client_thread_id is not None:
            thread_id = f'{thread_id}:{client_thread_id}'

        try:
            result = run_turn(question, thread_id=thread_id)
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


if __name__ == '__main__':
    app = create_app()
    if config.FLASK_DEBUG:
        app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=True, use_reloader=False)
    else:
        # waitress: pure-Python WSGI server, Windows-compatible (gunicorn is not).
        # Single worker process only -- LangGraph's in-memory session checkpointer
        # is per-process, so multiple worker processes would silently drop state.
        from waitress import serve
        print(f'Serving on http://{config.FLASK_HOST}:{config.FLASK_PORT} (waitress, single worker)')
        serve(app, host=config.FLASK_HOST, port=config.FLASK_PORT, threads=4)
