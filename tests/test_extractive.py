"""Decomposition, answer units and hybrid answering, with fake models."""
import core.extractive as ex

LIST_CHUNK = {'chunk_idx': 1, 'chunk_text': '[Discharge Medications] 1. Furosemide 60 mg PO daily '
                                            '2. Apixaban 5 mg PO BID 3. Potassium chloride 20 mEq PO daily'}
LAB_CHUNK = {'chunk_idx': 2, 'chunk_text': '[Pertinent Results] BNP 1850, creatinine 1.6. Sodium 134 today.'}


def test_decompose_carries_the_patient_into_each_part():
    parts = ex.decompose('What was the creatinine of the heart failure patient, '
                         'and which potassium supplement was prescribed at discharge?')
    assert parts == ['What was the creatinine of the heart failure patient?',
                     'which potassium supplement was prescribed at discharge for the heart failure patient?']
    assert ex.decompose('Which medications was the hip fracture patient discharged with?') == [
        'Which medications was the hip fracture patient discharged with?']


def test_information_need_drops_the_patient_description():
    assert ex.information_need('What was the TSH result for the patient with new onset atrial fibrillation?') \
        == 'What was the TSH result ?'


def test_numbered_lists_stay_whole():
    assert ex.answer_units(LIST_CHUNK['chunk_text']) == [LIST_CHUNK['chunk_text'].split('] ', 1)[1]]
    assert ex.answer_units(LAB_CHUNK['chunk_text']) == ['BNP 1850, creatinine 1.6.', 'Sodium 134 today.']


class Reranker:
    def __init__(self, score):
        self.score = score

    def predict(self, pairs):
        return [self.score] * len(pairs)


class Judge:
    """Answers the yes/no answerability prompt."""
    tokenizer = type('T', (), {'encode': lambda self, t, add_special_tokens=False: t.split(),
                               'decode': lambda self, t, skip_special_tokens=True: ' '.join(t)})()

    def __init__(self, reply):
        self.reply = reply

    def __call__(self, prompt, **_):
        return [{'generated_text': self.reply}]


def test_hybrid_returns_lists_verbatim_and_generates_other_parts(monkeypatch):
    import core.generation as gen
    monkeypatch.setattr(gen, 'generate_answer', lambda q, chunks, g: ('Creatinine was 1.6.', 0.0))
    answer, refused = ex.hybrid_answer([('creatinine?', [LAB_CHUNK]), ('supplement?', [LIST_CHUNK])],
                                       generator=None, judge=Judge('yes'), reranker=Reranker(5.0))
    assert not refused
    assert answer.startswith('Creatinine was 1.6.') and 'Potassium chloride 20 mEq' in answer


def test_hybrid_refuses_only_when_judge_and_score_agree(monkeypatch):
    import core.generation as gen
    monkeypatch.setattr(gen, 'generate_answer', lambda q, chunks, g: ('Something.', 0.0))
    parts = [('troponin?', [LAB_CHUNK])]
    assert ex.hybrid_answer(parts, None, Judge('no'), Reranker(-5.0), threshold=0.0) == (None, True)
    assert ex.hybrid_answer(parts, None, Judge('no'), Reranker(5.0), threshold=0.0)[1] is False
    assert ex.hybrid_answer(parts, None, Judge('yes'), Reranker(-5.0), threshold=0.0)[1] is False
