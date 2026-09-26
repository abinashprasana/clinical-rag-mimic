"""Generation scoring and the reflection observer hook. Fabricated text only;
the agent runs with stubbed retrieval and generation, so no models load."""
import uuid

from agent import graph
from evals.run_generation_eval import is_correct, outcome_of, run_agent, score

CHUNK = {'chunk_text': '[Discharge Medications] Furosemide 60 mg PO daily.',
         'subject_id': 9, 'hadm_id': 99, 'chunk_idx': 3, 'score': 0.9}


def q(qid, behavior, category, must=(), must_not=()):
    return {'id': qid, 'expected_behavior': behavior, 'category': category, 'question': 'x',
            'must_contain': list(must), 'must_not_contain': list(must_not)}


def test_outcomes():
    assert outcome_of({'route': 'clarify', 'needs_clarification': True}) == 'clarified'
    assert outcome_of({'route': 'dosage', 'fda_result': {'drug_name': 'x'}}) == 'fda_card'
    assert outcome_of({'route': 'retrieve', 'final_answer': graph.REFUSAL_TEXT}) == 'refused_by_gate'
    assert outcome_of({'route': 'retrieve',
                       'final_answer': 'I cannot find this information in the provided notes.'}) == 'model_refusal'
    assert outcome_of({'route': 'retrieve', 'final_answer': 'Furosemide 60 mg daily.'}) == 'shown'


def test_is_correct_per_behaviour():
    answer = q('q001', 'answer', 'medication_list', ['furosemide 60 mg'], ['furosemide 40 mg'])
    shown = {'outcome': 'shown', 'route': 'retrieve', 'final_answer': 'Furosemide 60 mg daily.'}
    assert is_correct(answer, shown)
    assert not is_correct(answer, {**shown, 'final_answer': 'Furosemide 40 mg, then 60 mg.'})
    assert not is_correct(answer, {**shown, 'outcome': 'refused_by_gate'})
    refuse = q('q002', 'refuse', 'unanswerable')
    assert is_correct(refuse, {'outcome': 'model_refusal', 'route': 'retrieve', 'final_answer': ''})
    assert not is_correct(refuse, shown)
    assert is_correct(q('q003', 'clarify', 'ambiguous'), {'outcome': 'clarified', 'route': 'clarify'})
    assert is_correct(q('q004', 'fda_lookup', 'fda_dosage'), {'outcome': 'fda_card', 'route': 'dosage'})
    assert not is_correct(q('q004', 'fda_lookup', 'fda_dosage'), shown)


def test_score_aggregates_with_intervals():
    questions = [q('q001', 'answer', 'single_fact', ['60 mg']), q('q002', 'refuse', 'unanswerable')]
    base = {'first_draft': 'd', 'first_supported': True, 'retried': False, 'retry_draft_changed': False,
            'latency_s': 1.0, 'route': 'retrieve'}
    records = [
        {**base, 'id': 'q001', 'category': 'single_fact', 'expected_behavior': 'answer',
         'outcome': 'shown', 'final_answer': 'Furosemide 60 mg.'},
        {**base, 'id': 'q002', 'category': 'unanswerable', 'expected_behavior': 'refuse',
         'outcome': 'shown', 'final_answer': 'Something else.'},
    ]
    result = score(questions, records)
    assert result['headline']['answer_correctness']['k'] == 1
    assert result['headline']['refusal_accuracy']['k'] == 0
    low, high = result['headline']['answer_correctness']['low'], result['headline']['answer_correctness']['high']
    assert 0 < low < 1 and high == 1.0
    assert result['gate']['outcomes'] == {'shown': 2}


def stub_graph(monkeypatch, draft):
    monkeypatch.setattr(graph.config, 'ANSWER_MODE', 'generative')
    monkeypatch.setattr(graph, 'retrieve_chunks', lambda *a, **k: [CHUNK])
    monkeypatch.setattr(graph, 'generate_answer', lambda question, chunks, gen: (draft, 0.0))
    monkeypatch.setattr(graph.llm, 'gemini_route', lambda *a, **k: None)
    return graph.build_graph(None, None, [CHUNK['chunk_text']], [{}], None)


def invoke(compiled, question):
    return compiled.invoke({
        'question': question, 'step_count': 0, 'reflection_regenerated': False,
        'needs_clarification': False, 'route': None, 'retrieved_chunks': [], 'fda_result': None,
    }, config={'configurable': {'thread_id': str(uuid.uuid4())}})


def comparable(state):
    return {k: v for k, v in state.items() if k != 'messages'}


def test_observer_hook_does_not_change_agent_output(monkeypatch):
    compiled = stub_graph(monkeypatch, 'The patient takes furosemide 60 mg daily.')
    question = 'What discharge medications were given?'
    without = invoke(compiled, question)
    seen = []
    graph._reflection_observers.append(lambda *args: seen.append(args))
    try:
        with_observer = invoke(compiled, question)
    finally:
        graph._reflection_observers.clear()
    assert comparable(with_observer) == comparable(without)
    assert seen and seen[0][3] == 1 and seen[0][2]['supported'] is True


def test_run_agent_records_first_decision_and_retry(monkeypatch):
    compiled = stub_graph(monkeypatch, 'The patient underwent cardiac bypass surgery.')
    records = run_agent([q('q001', 'answer', 'single_fact') | {'question': 'What surgery was done?'}], compiled)
    record = records[0]
    assert record['first_supported'] is False
    assert record['retried'] is True and record['retry_draft_changed'] is False
    assert record['outcome'] == 'refused_by_gate'
    assert record['passages'] == [3]
    assert graph._reflection_observers == []
