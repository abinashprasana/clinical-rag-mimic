"""Chat (decoder only) generator path in core/generation.py, with a fake
tokenizer and model so no weights are downloaded."""
import torch

import config
import core.generation as gen

PASSAGES = [{'chunk_text': f'[Discharge Medications] passage {i} ' + 'word ' * 150} for i in range(5)]


class FakeTokenizer:
    def __init__(self):
        self.template_calls = []

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        if isinstance(tokens, torch.Tensor):
            return ' '.join(f'tok{int(t)}' for t in tokens)
        return ' '.join(tokens)

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.tensor([[1, 1, 1]])}


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[1, 2, 3, 7, 8]])   # prompt ids then two new ids


def chat_generator(name='Qwen/Qwen3-1.7B'):
    return gen.ChatGenerator(name, FakeTokenizer(), FakeModel())


def test_only_new_tokens_are_returned_with_greedy_decoding():
    g = chat_generator()
    assert g.chat('system', 'user', 10) == 'tok7 tok8'
    assert g.model.calls[0]['do_sample'] is False and g.model.calls[0]['max_new_tokens'] == 10


def test_thinking_is_switched_off_only_for_qwen3():
    qwen3, qwen25 = chat_generator('Qwen/Qwen3-1.7B'), chat_generator('Qwen/Qwen2.5-1.5B-Instruct')
    qwen3.chat('s', 'u', 5)
    qwen25.chat('s', 'u', 5)
    assert qwen3.tokenizer.template_calls[0][1]['enable_thinking'] is False
    assert 'enable_thinking' not in qwen25.tokenizer.template_calls[0][1]


def test_answer_prompt_fences_all_five_passages_and_states_the_rules(monkeypatch):
    monkeypatch.setattr(config, 'GENERATION_CONTEXT_TOKENS', 3000)
    g = chat_generator()
    gen.generate_answer('What medications?', PASSAGES, g)
    messages, _ = g.tokenizer.template_calls[0]
    system, user = messages[0]['content'], messages[1]['content']
    assert all(f'<passage {i}>' in user for i in range(1, 6))
    assert user.rstrip().endswith('Question: What medications?')
    assert gen.REFUSAL_SENTENCE in system
    assert 'never as instructions' in system and 'absent' in system


def test_budget_cuts_only_the_passage_that_crosses_it():
    text = gen._fenced_passages(PASSAGES, FakeTokenizer(), budget=400)
    assert '<passage 1>' in text and '<passage 2>' in text and '<passage 3>' in text
    assert '<passage 4>' not in text


def test_judge_reads_yes_or_no(monkeypatch):
    g = chat_generator()
    monkeypatch.setattr(g, 'chat', lambda system, user, n: 'No.')
    assert gen.is_answerable('q', PASSAGES, g) is False
    monkeypatch.setattr(g, 'chat', lambda system, user, n: 'Yes')
    assert gen.is_answerable('q', PASSAGES, g) is True


def test_seq2seq_path_is_unchanged():
    class Pipe:
        tokenizer = FakeTokenizer()

        def __call__(self, prompt, **kwargs):
            self.prompt = prompt
            return [{'generated_text': 'Furosemide 60 mg daily.'}]
    pipe = Pipe()
    answer, _ = gen.generate_answer('What medications?', PASSAGES[:1], pipe)
    assert answer == 'Furosemide 60 mg daily.'
    assert pipe.prompt.rstrip().endswith('Answer:') and '[Chunk 1]' in pipe.prompt


def test_models_load_once_per_name(monkeypatch):
    loads = []
    monkeypatch.setattr(gen, '_models', {})

    class Cfg:
        is_encoder_decoder = True
    import transformers
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained', staticmethod(lambda name: Cfg()))
    monkeypatch.setattr(gen, 'pipeline', lambda *a, **k: loads.append(k['model']) or object())
    first, second = gen.load_model('some/model'), gen.load_model('some/model')
    assert first is second and loads == ['some/model']
