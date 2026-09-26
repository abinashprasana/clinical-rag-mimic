"""UI summary: built only from reviewed aggregate files, validated on load,
and importable without numpy or torch (the public Vercel function imports it)."""
import json
import subprocess
import sys

from evals.ui_summary import build_summary, load_summary, valid, write_summary

RATE = {'k': 3, 'n': 4, 'rate': 0.75, 'low': 0.3, 'high': 0.95}
CI = {'mean': 0.7, 'low': 0.5, 'high': 0.9}


def write(folder, name, payload):
    (folder / name).write_text(json.dumps(payload), encoding='utf-8')


def retrieval(reviewed_only=True):
    row = {m: CI for m in ('recall@5', 'mrr', 'ndcg@5')} | {'latency_p50_ms': 10.0, 'latency_p95_ms': 12.0}
    return {'meta': {'reviewed_only': reviewed_only, 'n_questions': 30, 'run_at': 't'},
            'overall': {'dense': row, 'bm25': row | {'paired_vs_dense': {'recall@5': CI}}}}


def test_missing_results_give_an_all_pending_summary(tmp_path):
    summary = build_summary(str(tmp_path), 'demo')
    assert valid(summary)
    assert all(summary[s] is None for s in ('retrieval', 'generation', 'faithfulness', 'stress', 'injection'))


def test_unreviewed_results_never_reach_the_summary(tmp_path):
    write(tmp_path, 'retrieval_demo.json', retrieval(reviewed_only=False))
    assert build_summary(str(tmp_path), 'demo')['retrieval'] is None
    write(tmp_path, 'retrieval_demo.json', retrieval())
    summary = build_summary(str(tmp_path), 'demo')
    assert [v['name'] for v in summary['retrieval']['variants']] == ['dense', 'bm25']
    assert summary['retrieval']['variants'][1]['diff_recall@5'] == CI


def test_summary_carries_no_text_or_question_ids(tmp_path):
    write(tmp_path, 'generation_demo.json', {
        'meta': {'reviewed_only': True, 'n_questions': 9, 'run_at': 't'},
        'scores': {'headline': {'answer_correctness': RATE, 'refusal_accuracy': RATE, 'routing_accuracy': RATE}},
        'legacy': {'source': 'secret text should not travel'},
    })
    path = tmp_path / 'summary.json'
    write_summary(str(tmp_path), 'demo', str(path))
    text = path.read_text(encoding='utf-8')
    assert 'secret text' not in text and 'q0' not in text
    assert load_summary(str(path))['generation']['answer_correctness'] == RATE


def test_load_summary_rejects_missing_and_malformed_files(tmp_path):
    assert load_summary(str(tmp_path / 'absent.json')) is None
    (tmp_path / 'bad.json').write_text('{not json', encoding='utf-8')
    assert load_summary(str(tmp_path / 'bad.json')) is None
    write(tmp_path, 'wrong.json', {'schema': 1, 'corpus': 'demo', 'generation': {'answer_correctness': {'k': 1}}})
    assert load_summary(str(tmp_path / 'wrong.json')) is None
    write(tmp_path, 'extra.json', {'schema': 1, 'corpus': 'demo', 'answers': ['text']})
    assert load_summary(str(tmp_path / 'extra.json')) is None


def test_committed_demo_summary_is_valid():
    assert load_summary('evals/results/summary_demo.json') is not None


def test_loader_imports_no_heavy_packages():
    code = ('import sys, evals.ui_summary; '
            'print(",".join(m for m in ("numpy", "torch", "faiss", "sentence_transformers") if m in sys.modules))')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ''
