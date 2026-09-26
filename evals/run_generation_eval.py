"""Generation evaluation through the real LangGraph agent, scored separately
from retrieval.

    python -m evals.run_generation_eval --corpus demo
    python -m evals.run_generation_eval --corpus demo --include-unreviewed   # development only
    python -m evals.run_generation_eval --corpus real                        # local, writes outputs/eval/

What is scored
* answer questions: correct when the final answer is shown, contains every
  must_contain term and none of the must_not_contain terms.
* refuse questions: correct when the final answer is a refusal, either the
  model's own "cannot find this information" or the faithfulness gate's.
* clarify and fda_lookup questions: correct when the agent takes that branch.

Every run is offline and deterministic: Gemini is disabled, so routing uses
the local keyword rules, and the openFDA lookup is replaced by a stub (only
the routing decision is scored). The faithfulness gate's first decision and
the final outcome are recorded per question through the observer hook in
agent/graph.py, which does not change what the agent returns.
"""
import argparse
import json
import os
import sys
import time
import uuid
from collections import Counter

from evals import common
from evals.corpus import load_corpus
from evals.golden import GOLDEN_FILES, contains, load_questions, select, validate
from evals.metrics import wilson_ci

MODEL_REFUSAL = 'cannot find this information'
FDA_STUB = {'drug_name': 'stub', 'source': 'evaluation stub (no network call)'}


def offline_agent_patches():
    """Disable Gemini and stub openFDA for the whole process."""
    import config
    from agent import graph, llm
    config.GEMINI_API_KEY = ''
    llm._client, llm._client_checked = None, True
    graph.fda_label_tool = lambda name: {**FDA_STUB, 'drug_name': name}


def outcome_of(state):
    from agent.graph import REFUSAL_TEXT
    final = state.get('final_answer') or ''
    if state.get('route') == 'clarify' or state.get('needs_clarification'):
        return 'clarified'
    if state.get('fda_result'):
        return 'fda_card'
    if final == REFUSAL_TEXT:
        return 'refused_by_gate'
    if MODEL_REFUSAL in final.lower():
        return 'model_refusal'
    return 'shown'


def is_correct(question, record):
    behavior, outcome = question['expected_behavior'], record['outcome']
    if behavior == 'answer':
        answer = record['final_answer'] or ''
        return (
            outcome == 'shown'
            and all(contains(answer, t) for t in question['must_contain'])
            and not any(contains(answer, t) for t in question['must_not_contain'])
        )
    if behavior == 'refuse':
        return outcome in ('refused_by_gate', 'model_refusal')
    if behavior == 'clarify':
        return record['route'] == 'clarify'
    if behavior == 'fda_lookup':
        return record['route'] == 'dosage'
    raise ValueError(behavior)


def run_agent(questions, graph):
    """One fresh thread per question, so no conversation memory leaks between them."""
    import agent.graph as agent_graph
    reflections = []
    observer = lambda q, draft, reflection, attempt: reflections.append(
        {'attempt': attempt, 'draft': draft, **reflection})
    agent_graph._reflection_observers.append(observer)
    records = []
    try:
        for q in questions:
            reflections.clear()
            start = time.perf_counter()
            state = graph.invoke({
                'question': q['question'], 'step_count': 0, 'reflection_regenerated': False,
                'needs_clarification': False, 'route': None, 'retrieved_chunks': [], 'fda_result': None,
            }, config={'configurable': {'thread_id': str(uuid.uuid4())}})
            first = reflections[0] if reflections else None
            records.append({
                'id': q['id'],
                'category': q['category'],
                'expected_behavior': q['expected_behavior'],
                'route': state.get('route'),
                'passages': [c['chunk_idx'] for c in state.get('retrieved_chunks', [])],
                'first_draft': first['draft'] if first else None,
                'first_supported': first['supported'] if first else None,
                'first_unsupported_claims': first['unsupported_claims'] if first else [],
                'retried': len(reflections) > 1,
                'retry_draft_changed': len(reflections) > 1 and reflections[1]['draft'] != reflections[0]['draft'],
                'final_answer': state.get('final_answer'),
                'outcome': outcome_of(state),
                'latency_s': time.perf_counter() - start,
            })
    finally:
        agent_graph._reflection_observers.remove(observer)
    return records


def rate(flags):
    k, n = sum(flags), len(flags)
    p, low, high = wilson_ci(k, n)
    return {'k': k, 'n': n, 'rate': p, 'low': low, 'high': high}


def score(questions, records):
    by_id = {q['id']: q for q in questions}
    for r in records:
        r['correct'] = is_correct(by_id[r['id']], r)

    def pick(pred):
        return [r['correct'] for r in records if pred(r)]

    headline = {
        'answer_correctness': rate(pick(lambda r: r['expected_behavior'] == 'answer')),
        'refusal_accuracy': rate(pick(lambda r: r['expected_behavior'] == 'refuse')),
        'routing_accuracy': rate(pick(lambda r: r['expected_behavior'] in ('clarify', 'fda_lookup'))),
        'routing_clarify': rate(pick(lambda r: r['expected_behavior'] == 'clarify')),
        'routing_fda': rate(pick(lambda r: r['expected_behavior'] == 'fda_lookup')),
    }
    categories = sorted({r['category'] for r in records})
    per_category = {c: rate(pick(lambda r, c=c: r['category'] == c)) for c in categories}
    judged = [r for r in records if r['first_supported'] is not None]
    gate = {
        'drafts_checked': len(judged),
        'first_check_passed': sum(r['first_supported'] for r in judged),
        'retried': sum(r['retried'] for r in records),
        'retry_changed_draft': sum(r['retry_draft_changed'] for r in records),
        'outcomes': dict(Counter(r['outcome'] for r in records)),
        'answerable_refused_by_gate': sum(
            r['outcome'] == 'refused_by_gate' for r in records if r['expected_behavior'] == 'answer'),
    }
    latencies = sorted(r['latency_s'] for r in records)
    return {'headline': headline, 'per_category': per_category, 'gate': gate,
            'mean_latency_s': sum(latencies) / len(latencies) if latencies else float('nan')}


def legacy_demo(embed_model, index, chunks, provenance, generator):
    """Fresh run of the original 10 question keyword check (demo/evaluate.py's
    method) with the models already loaded."""
    from core.generation import generate_answer
    from core.retrieval import retrieve_chunks
    from demo.evaluate import EVAL_QUESTIONS, KEYWORD_ALTERNATIVES
    hits = []
    for question, keyword in EVAL_QUESTIONS:
        retrieved = retrieve_chunks(question, embed_model, index, chunks, provenance, k=5)
        answer, _ = generate_answer(question, retrieved, generator)
        hits.append(any(alt in answer.lower() for alt in KEYWORD_ALTERNATIVES.get(keyword, [keyword])))
    return {**rate(hits), 'source': 'rerun now with demo/evaluate.py questions and keywords'}


def legacy_real():
    """The real legacy row is read from the existing local results file; it
    is never rerun here because core/evaluation.py rewrites charts."""
    path = os.path.join('outputs', 'evaluation_results.json')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        results = json.load(f)
    run_on = time.strftime('%Y-%m-%d', time.localtime(os.path.getmtime(path)))
    return {**rate([bool(r['keyword_found']) for r in results]),
            'source': f'existing outputs/evaluation_results.json from {run_on}, not rerun'}


def pct(row):
    if not row or not row['n']:
        return 'n/a'
    return f"{row['k']}/{row['n']} = {row['rate']:.0%} [{row['low']:.0%}, {row['high']:.0%}]"


def render(result, fmt):
    table = common.markdown_table if fmt == 'md' else common.text_table
    parts = []
    if not result['meta']['reviewed_only']:
        parts.append(common.UNREVIEWED_BANNER)
    parts.append(f"Corpus {result['meta']['corpus']}, {result['meta']['n_questions']} questions. "
                 'Rates with 95% Wilson intervals. Routing is offline keyword rules (Gemini disabled); '
                 'openFDA is stubbed.')
    h = result['scores']['headline']
    rows = [
        ['Answer correctness (answer questions)', pct(h['answer_correctness'])],
        ['Refusal accuracy (unanswerable questions)', pct(h['refusal_accuracy'])],
        ['Routing accuracy (clarify and FDA questions)', pct(h['routing_accuracy'])],
        ['  of which clarify', pct(h['routing_clarify'])],
        ['  of which FDA label lookup', pct(h['routing_fda'])],
    ]
    if result.get('legacy'):
        rows.append(['Original smoke test (10 questions, keyword match)', pct(result['legacy'])])
    parts.append(table(['measure', 'result'], rows))
    parts.append(table(['category', 'correct'],
                       [[c, pct(r)] for c, r in result['scores']['per_category'].items()]))
    g = result['scores']['gate']
    parts.append(table(['faithfulness gate', 'count'], [
        ['drafts checked', g['drafts_checked']],
        ['passed on first check', g['first_check_passed']],
        ['retried', g['retried']],
        ['retry produced a different draft', g['retry_changed_draft']],
        ['answerable questions refused by the gate', g['answerable_refused_by_gate']],
        *[[f'outcome: {k}', v] for k, v in sorted(g['outcomes'].items())],
    ]))
    if result.get('legacy'):
        parts.append(f"Original smoke test source: {result['legacy']['source']}.")
    return '\n\n'.join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--corpus', choices=['demo', 'real'], default='demo')
    parser.add_argument('--include-unreviewed', action='store_true',
                        help='also score draft questions (development only, labelled UNREVIEWED)')
    parser.add_argument('--no-legacy', action='store_true', help='skip the original 10 question row')
    parser.add_argument('--file', help='override the golden file path')
    args = parser.parse_args(argv)
    common.guard_real_data(args.corpus)

    path = args.file or GOLDEN_FILES[args.corpus]
    if not os.path.exists(path):
        print(f'{path} does not exist. See evals/golden/README_real_questions.md.')
        return 2
    corpus = load_corpus(args.corpus)
    all_questions = load_questions(path)
    errors, _ = validate(all_questions, corpus)
    if errors:
        print(f'{len(errors)} validation errors; run python -m evals.golden --corpus {args.corpus}')
        return 1
    questions = select(all_questions, args.include_unreviewed)
    if not questions:
        print('No reviewed questions. Mark questions "reviewed": true, '
              'or pass --include-unreviewed for a development run.')
        return 2

    offline_agent_patches()
    import faiss
    from sentence_transformers import SentenceTransformer

    import config
    from agent.graph import build_graph
    from core.generation import load_generator
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    index = faiss.read_index(os.path.join(corpus.output_dir, 'faiss_index.index'))
    generator = load_generator()
    graph = build_graph(embed_model, index, corpus.chunks, corpus.provenance, generator)

    records = run_agent(questions, graph)
    result = {
        'meta': common.run_metadata(args.corpus, args.include_unreviewed, len(questions)),
        'generator': config.LOCAL_GENERATOR_MODEL,
        'routing': 'offline keyword rules; openFDA stubbed',
        'scores': score(questions, records),
    }
    if not args.no_legacy:
        result['legacy'] = (legacy_demo(embed_model, index, corpus.chunks, corpus.provenance, generator)
                            if args.corpus == 'demo' else legacy_real())

    agg_dir, records_dir = common.output_dirs(args.corpus, args.include_unreviewed)
    stem = f'generation_{args.corpus}' + ('.unreviewed' if args.include_unreviewed else '')
    common.write_json(os.path.join(agg_dir, stem + '.json'), result)
    with open(os.path.join(agg_dir, stem + '.md'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(render(result, 'md') + '\n')
    common.write_jsonl(os.path.join(records_dir, f'generation_records_{args.corpus}.jsonl'), records)
    print(render(result, 'text'))
    common.refresh_summary(args.corpus, args.include_unreviewed)
    print(f'\nWrote {agg_dir}/{stem}.json and .md; per question records in {records_dir}/.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
