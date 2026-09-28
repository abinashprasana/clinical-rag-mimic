"""Local fixture server for browser and accessibility checks."""

from pathlib import Path
import uuid

from flask import Flask, jsonify, render_template, request, session


ROOT = Path(__file__).resolve().parents[1]
app = Flask(
    __name__,
    template_folder=str(ROOT / 'templates'),
    static_folder=str(ROOT / 'static'),
    static_url_path='/static',
)
app.secret_key = 'ui-fixture-only'


@app.after_request
def add_security_headers(response):
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; base-uri 'self'; object-src 'none'; "
        "frame-ancestors 'none'; form-action 'self'; connect-src 'self'; "
        "img-src 'self'; font-src 'self'; script-src 'self'; style-src 'self'"
    )
    return response


def _rate(k, n, low, high):
    return {'k': k, 'n': n, 'rate': k / n, 'low': low, 'high': high}


def _interval(mean, low, high):
    return {'mean': mean, 'low': low, 'high': high}


# Fixture values for layout tests only. They are not measurements and never
# leave this test server.
FIXTURE_EVAL_SUMMARY = {
    'schema': 1,
    'corpus': 'demo',
    'retrieval': {
        'n': 30, 'default': 'dense', 'run_at': '2026-01-01T00:00:00+00:00',
        'variants': [
            {'name': 'dense', 'recall@5': _interval(0.7, 0.55, 0.84), 'mrr': _interval(0.6, 0.45, 0.74),
             'ndcg@5': _interval(0.58, 0.44, 0.71), 'diff_recall@5': None, 'p50_ms': 11.0, 'p95_ms': 15.0},
            {'name': 'dense_legacy', 'recall@5': _interval(0.5, 0.35, 0.65), 'mrr': _interval(0.4, 0.27, 0.54),
             'ndcg@5': _interval(0.42, 0.29, 0.55), 'diff_recall@5': _interval(-0.2, -0.36, -0.05),
             'p50_ms': 11.0, 'p95_ms': 15.0},
            {'name': 'bm25', 'recall@5': _interval(0.4, 0.25, 0.55), 'mrr': _interval(0.3, 0.17, 0.45),
             'ndcg@5': _interval(0.3, 0.18, 0.44), 'diff_recall@5': _interval(-0.3, -0.47, -0.15),
             'p50_ms': 1.0, 'p95_ms': 1.0},
        ],
    },
    'generation': {
        'n': 45, 'run_at': '2026-01-01T00:00:00+00:00',
        'answer_correctness': _rate(18, 30, 0.42, 0.75),
        'refusal_accuracy': _rate(3, 8, 0.14, 0.69),
        'routing_accuracy': _rate(6, 7, 0.49, 0.97),
    },
    'faithfulness': None,
    'stress': None,
    'injection': {
        'questions': 9, 'run_at': '2026-01-01T00:00:00+00:00',
        'paths': [{'name': 'flan-t5 (local agent)', 'canary_final': _rate(1, 6, 0.03, 0.56),
                   'gate_passed': _rate(1, 1, 0.21, 1.0)}],
    },
}


@app.get('/')
def index():
    public_demo = request.args.get('mode', 'local') == 'public'
    eval_demo = None if request.args.get('eval') == 'pending' else FIXTURE_EVAL_SUMMARY
    return render_template(
        'index.html',
        accuracy_pct=100 if public_demo else 80,
        accuracy_label='10/10 canonical checks' if public_demo else None,
        dataset_version=(
            'Synthetic demo (fabricated notes; no patient data; real '
            'MIMIC-IV-Note data requires credentialed PhysioNet access)'
            if public_demo
            else 'MIMIC-IV-Note v2.2'
        ),
        retrieval_top_k=5,
        generator_model=(
            'gemini-flash-lite-latest (retrieval-augmented, local fallback)'
            if public_demo
            else 'google/flan-t5-base'
        ),
        public_demo=public_demo,
        gemini_enabled=public_demo,
        eval_demo=eval_demo,
        eval_real=None,
    )


def evidence_passages():
    return [
        {
            'subject_id': 100000 + index,
            'hadm_id': 200000 + index,
            'chunk_idx': index,
            'score': round(0.93 - index * 0.06, 4),
            'chunk_text': (
                f'Passage content {index + 1}. The discharge summary records '
                'diagnosis, medication, and follow-up information for review.'
            ),
        }
        for index in range(5)
    ]


@app.post('/ask')
def ask():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get('question', '')).strip()
    thread_id = payload.get('thread_id')
    if not question:
        return jsonify({'error': {'code': 'INVALID_QUESTION', 'message': 'Please enter a valid clinical question.'}}), 400
    try:
        uuid.UUID(str(thread_id))
    except ValueError:
        return jsonify({'error': {'code': 'REQUEST_FAILED', 'message': 'We could not complete this request. Please try again.'}}), 400

    session.setdefault('session_id', str(uuid.uuid4()))
    lowered = question.lower()

    if 'server error' in lowered:
        return jsonify({'error': {'code': 'REQUEST_FAILED', 'message': 'We could not complete this request. Please try again.'}}), 500

    if 'clarify' in lowered:
        return jsonify({
            'answer': 'Which admission or date should I review?',
            'route': 'clarify',
            'tool_used': 'Clarification requested',
            'citations': [],
            'reflection': None,
            'needs_clarification': True,
            'fda_result': None,
        })

    if 'metformin' in lowered or 'fda' in lowered:
        return jsonify({
            'answer': None,
            'route': 'dosage',
            'tool_used': 'FDA Label Lookup: metformin',
            'citations': [],
            'reflection': None,
            'needs_clarification': False,
            'fda_result': {
                'drug_name': 'metformin',
                'matched_brand_name': 'Metformin hydrochloride',
                'matched_generic_name': 'metformin hydrochloride',
                'dosage_and_administration': 'Individualize dosing on the basis of effectiveness and tolerability.',
                'contraindications': 'Contraindicated in patients with severe renal impairment.',
                'warnings': 'Review the complete label for warnings, precautions, and monitoring information.',
                'source': 'openFDA Drug Label API',
                'label_url': 'https://labels.fda.gov/?search=metformin',
            },
        })

    unsupported = 'unsupported' in lowered
    return jsonify({
        'answer': (
            'I could not confirm this answer is fully supported by the retrieved evidence. '
            'Please review the source passages or refine the question.'
            if unsupported
            else 'The retrieved discharge summary documents an acute respiratory diagnosis. '
                 'It also records discharge medication instructions and follow-up with the clinical team. '
                 'Review the evidence passages for the exact record language.'
        ),
        'route': 'retrieve',
        'tool_used': 'Retrieved 5 evidence passages',
        'citations': evidence_passages(),
        'reflection': {
            'supported': not unsupported,
            'unsupported_claims': ['Source overlap was insufficient.'] if unsupported else [],
        },
        'needs_clarification': False,
        'fda_result': None,
    })


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5055, debug=False, use_reloader=False)
