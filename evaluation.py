import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sentence_transformers import SentenceTransformer

from retrieval import load_index, retrieve_chunks
from generation import load_generator, generate_answer

def main():
    print('Loading models and index for evaluation...')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index_loaded, chunks_loaded, _ = load_index('outputs/')
    generator = load_generator()

    eval_questions = [
        ('What is diabetes?',                                              'diabetes mellitus'),
        ('What medications are used for blood glucose control?',           'insulin'),
        ('What is HbA1c?',                                                 'glycated hemoglobin'),
        ('What does BID mean?',                                            'twice daily'),
        ('What are signs of hypoglycemia?',                                'low blood sugar'),
        ('What diet is recommended for diabetic patients?',                'low carbohydrate'),
        ('What is metformin used for?',                                    'diabetes'),
        ('What does PRN mean?',                                            'as needed'),
        ('What is the normal blood glucose range?',                        'glucose'),
        ('What follow-up is recommended after discharge?',                 'primary care'),
    ]

    keyword_alternatives = {
        'diabetes mellitus':  ['diabetes mellitus', 'diabetes', 'diabetic'],
        'insulin':            ['insulin'],
        'glycated hemoglobin':['glycated hemoglobin', 'hemoglobin a1c', 'hba1c', 'glycosylated'],
        'twice daily':        ['twice daily', 'twice a day', 'two times a day', 'bid'],
        'low blood sugar':    ['low blood sugar', 'hypoglycemia', 'hypoglycaemia',
                               'low glucose', 'sweating', 'shakiness', 'dizziness',
                               'confusion', 'weakness', 'blood sugar'],
        'low carbohydrate':   ['low carbohydrate', 'low carb', 'carbohydrate',
                               'diabetic diet', 'healthy diet', 'diet', 'nutrition',
                               'food', 'eating'],
        'diabetes':           ['diabetes', 'diabetic', 'blood glucose'],
        'as needed':          ['as needed', 'as required', 'when necessary',
                               'when needed', 'if needed', 'when required', 'prn'],
        'glucose':            ['glucose', 'blood sugar', 'blood glucose',
                               'mg/dl', 'mmol', '70', '80', '100', 'normal range',
                               'fasting', 'sugar level'],
        'primary care':       ['primary care', 'follow up', 'follow-up',
                               'outpatient', 'physician', 'doctor', 'clinic'],
    }

    results = []
    print('=== Evaluation Run ===\n')
    print(f'{"Question":<60} {"Keyword":<25} {"Hit":<5} {"Time (s)":<8}')
    print('-' * 105)

    for question, keyword in eval_questions:
        chunks  = retrieve_chunks(question, model, index_loaded, chunks_loaded, k=5)
        answer, latency  = generate_answer(question, chunks, generator)
        latency = round(latency, 2)

        alternatives = keyword_alternatives.get(keyword, [keyword])
        hit = any(alt.lower() in answer.lower() for alt in alternatives)

        results.append({
            'question':         question,
            'answer':           answer,
            'expected_keyword': keyword,
            'keyword_found':    hit,
            'latency_seconds':  latency
        })

        tick = '\u2713' if hit else '\u2717'
        print(f'{question:<60} {keyword:<25} {tick:<5} {latency:<8}')

    accuracy = sum(r['keyword_found'] for r in results)
    avg_lat  = round(sum(r['latency_seconds'] for r in results) / len(results), 2)

    print('-' * 105)
    print(f'\nAccuracy: {accuracy} / {len(results)}  ({accuracy * 10:.0f}%)')
    print(f'Average latency: {avg_lat}s per question')

    os.makedirs('outputs', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    with open('outputs/evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    questions_short = [q['question'][:42] + '...' if len(q['question']) > 42 else q['question']
                       for q in results]
    hits    = [int(r['keyword_found']) for r in results]
    latencies = [r['latency_seconds'] for r in results]
    colors  = ['#4CAF50' if h else '#E57373' for h in hits]

    # Accuracy Chart
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.barh(questions_short, hits, color=colors, edgecolor='white')
    ax.set_xlim(0, 1.3)
    ax.set_xlabel('Keyword found (1 = Yes, 0 = No)')
    ax.set_title(f'Evaluation — Keyword Accuracy ({accuracy}/{len(results)} correct)')
    hit_patch  = mpatches.Patch(color='#4CAF50', label='Keyword found')
    miss_patch = mpatches.Patch(color='#E57373', label='Keyword not found')
    ax.legend(handles=[hit_patch, miss_patch], loc='lower right')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('static/eval_accuracy.png', dpi=120, bbox_inches='tight')
    plt.close()

    # Latency Chart
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.barh(questions_short, latencies, color='steelblue', edgecolor='white')
    ax.axvline(avg_lat, color='tomato', linestyle='--', linewidth=1.5,
               label=f'Mean latency: {avg_lat}s')
    ax.set_xlabel('Latency (seconds)')
    ax.set_title('Per-Question Latency')
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('static/eval_latency.png', dpi=120, bbox_inches='tight')
    plt.close()
    
    print('Evaluation results and charts saved.')

if __name__ == '__main__':
    main()
