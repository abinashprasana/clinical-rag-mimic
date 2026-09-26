"""Answerability check and passage count, with a fake generator (no model)."""
import uuid

import config
from agent import graph
from core.generation import build_prompt, is_answerable

CHUNKS = [{'chunk_text': '[Discharge Medications] Furosemide 60 mg PO daily.', 'subject_id': 1,
           'hadm_id': 10, 'chunk_idx': i, 'score': 0.9} for i in range(5)]


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return ' '.join(tokens)


class FakeGenerator:
    tokenizer = FakeTokenizer()

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def __call__(self, prompt, **_):
        self.prompts.append(prompt)
        return [{'generated_text': self.reply}]


def test_answerability_prompt_and_decision():
    yes, no = FakeGenerator('yes'), FakeGenerator('no')
    assert is_answerable('What is the troponin?', CHUNKS, yes) is True
    assert is_answerable('What is the troponin?', CHUNKS, no) is False
    assert 'Answer yes or no.' in no.prompts[0] and 'Furosemide 60 mg' in no.prompts[0]
    assert is_answerable('q', CHUNKS, FakeGenerator('')) is True   # only an explicit no refuses


def test_build_prompt_default_is_unchanged():
    assert build_prompt('Q?', ['ctx']).endswith('Question: Q?\n\nAnswer:')


def run(monkeypatch, check, reply, top_k=2):
    seen = {}
    monkeypatch.setattr(config, 'ANSWER_MODE', 'generative')
    monkeypatch.setattr(config, 'ANSWERABILITY_CHECK', check)
    monkeypatch.setattr(config, 'GENERATION_TOP_K', top_k)
    monkeypatch.setattr(graph, 'retrieve_chunks', lambda *a, **k: CHUNKS)
    monkeypatch.setattr(graph.llm, 'gemini_route', lambda *a, **k: None)
    monkeypatch.setattr(graph, 'is_answerable', lambda q, passages, gen: reply)

    def fake_generate(question, passages, gen):
        seen['n'] = len(passages)
        return 'Furosemide 60 mg daily.', 0.0
    monkeypatch.setattr(graph, 'generate_answer', fake_generate)
    compiled = graph.build_graph(None, None, [], [], None)
    state = compiled.invoke({'question': 'What discharge medications were given?', 'step_count': 0,
                             'reflection_regenerated': False, 'needs_clarification': False, 'route': None,
                             'retrieved_chunks': [], 'fda_result': None},
                            config={'configurable': {'thread_id': str(uuid.uuid4())}})
    return state, seen


def test_generation_top_k_limits_prompt_passages_but_not_citations(monkeypatch):
    state, seen = run(monkeypatch, False, True, top_k=2)
    assert seen['n'] == 2 and len(state['citations']) == 5


def test_answerability_refusal_replaces_generation(monkeypatch):
    state, seen = run(monkeypatch, True, False)
    assert state['final_answer'] == graph.NOT_IN_NOTES_TEXT and 'n' not in seen
    state, seen = run(monkeypatch, True, True)
    assert state['final_answer'] == 'Furosemide 60 mg daily.'
