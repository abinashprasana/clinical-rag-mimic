"""Runs the same 10-question keyword-hit evaluation evaluation.py uses for
the real dataset, but against the synthetic demo index in outputs_demo/.
Writes outputs_demo/evaluation_results.json -- separate from the real
outputs/evaluation_results.json -- and never touches static/eval_*.png
(those charts are tied to the real dataset's README numbers).
"""
import json
import os

from sentence_transformers import SentenceTransformer

from retrieval import load_index, retrieve_chunks
from generation import load_generator, generate_answer
import config

DEMO_OUTPUT_DIR = 'outputs_demo'

EVAL_QUESTIONS = [
    ('What is the discharge diagnosis?', 'diagnosis'),
    ('What is the discharge condition?', 'stable'),
    ('What is the discharge disposition?', 'disposition'),
    ('What follow-up care is recommended after discharge?', 'primary care'),
    ('What medications are given at discharge?', 'medication'),
    ('What is documented in the past medical history?', 'history'),
    ('What allergies does the patient have?', 'allergy'),
    ('What vital signs were recorded?', 'vital signs'),
    ('What lab results were noted?', 'lab results'),
    ('What is the chief complaint?', 'chief complaint'),
]

KEYWORD_ALTERNATIVES = {
    'diagnosis':       ['diagnosis', 'diagnoses', 'diagnosed'],
    'stable':          ['stable', 'condition', 'improved', 'good', 'fair'],
    'disposition':     ['home', 'facility', 'rehab', 'rehabilitation',
                        'skilled nursing', 'extended care', 'discharged to'],
    'primary care':    ['primary care', 'follow up', 'follow-up',
                        'outpatient', 'physician', 'doctor', 'clinic', 'appointment'],
    'medication':      ['medication', 'mg', 'tablet', 'dose', 'daily', 'prescribed'],
    'history':         ['history', 'hypertension', 'diabetes', 'disease',
                        'condition', 'prior', 'chronic'],
    'allergy':         ['allergy', 'allergies', 'nkda', 'no known', 'penicillin', 'reaction'],
    'vital signs':     ['blood pressure', 'heart rate', 'temperature', 'pulse',
                        'respiratory rate', 'vital', 'oxygen saturation',
                        'bp:', 'hr:', 'rr:', 'o2 sat', 'temp:'],
    'lab results':     ['lab', 'result', 'blood', 'level', 'value', 'glucose', 'count',
                        'ast', 'alt', 'wbc', 'hgb', 'creatinine', 'sodium'],
    'chief complaint': ['complaint', 'presented', 'admitted for', 'admitted with', 'presenting'],
}


def main():
    print('Loading models and demo index for evaluation...')
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    idx, chunks, provenance = load_index(DEMO_OUTPUT_DIR)
    generator = load_generator()

    results = []
    print(f'{"Question":<52} {"Keyword":<16} {"Hit":<5} {"Time (s)":<8}')
    print('-' * 90)

    for question, keyword in EVAL_QUESTIONS:
        retrieved = retrieve_chunks(question, model, idx, chunks, provenance, k=config.DEFAULT_TOP_K)
        answer, latency = generate_answer(question, retrieved, generator)
        latency = round(latency, 2)

        alternatives = KEYWORD_ALTERNATIVES.get(keyword, [keyword])
        hit = any(alt.lower() in answer.lower() for alt in alternatives)

        results.append({
            'question': question, 'answer': answer, 'expected_keyword': keyword,
            'keyword_found': hit, 'latency_seconds': latency,
        })
        print(f'{question:<52} {keyword:<16} {"YES" if hit else "no":<5} {latency:<8}')

    accuracy = sum(r['keyword_found'] for r in results)
    avg_lat = round(sum(r['latency_seconds'] for r in results) / len(results), 2)
    print('-' * 90)
    print(f'\nAccuracy: {accuracy}/{len(results)} ({accuracy * 10:.0f}%)')
    print(f'Average latency: {avg_lat}s per question')

    with open(os.path.join(DEMO_OUTPUT_DIR, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f'Saved {DEMO_OUTPUT_DIR}/evaluation_results.json')


if __name__ == '__main__':
    main()
