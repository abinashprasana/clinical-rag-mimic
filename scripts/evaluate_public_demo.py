"""Measures the actual accuracy of the deployed Vercel demo runtime
(demo_runtime.py's deterministic extractive method), using the same
10-question keyword-hit methodology as evaluation.py and evaluate_demo.py.

This is a separate, real measurement from evaluate_demo.py: that script
measures the full local pipeline (SentenceTransformer/FAISS/FLAN-T5) running
on the synthetic notes, while this script measures the simpler extractive
method that is actually deployed to the public Vercel demo. The two use
different answering logic and should not be assumed to score the same.
"""
import json
import os
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import demo_runtime
import viz_style
from scripts.evaluate_demo import EVAL_QUESTIONS, KEYWORD_ALTERNATIVES

CHART_DIR = 'outputs_demo/ui_charts'


def _plot_charts(results):
    os.makedirs(CHART_DIR, exist_ok=True)
    viz_style.apply_style()

    questions = [
        item['question'][:42] + '...' if len(item['question']) > 42 else item['question']
        for item in results
    ]
    hits = [int(item['keyword_found']) for item in results]
    latencies = [float(item['latency_seconds']) for item in results]
    accuracy = sum(hits)

    _, ax = plt.subplots(figsize=(12, 4.5))
    colors = [viz_style.SUCCESS if hit else viz_style.DANGER for hit in hits]
    ax.barh(questions, hits, color=colors, edgecolor=viz_style.BORDER)
    ax.set_xlim(0, 1.3)
    ax.set_xlabel('Keyword found (1 = yes, 0 = no)')
    ax.set_title(f'Public Demo Keyword-Hit Evaluation ({accuracy}/{len(results)} questions)')
    ax.legend(
        handles=[
            mpatches.Patch(color=viz_style.SUCCESS, label='Keyword found'),
            mpatches.Patch(color=viz_style.DANGER, label='Keyword not found'),
        ],
        loc='lower right',
    )
    ax.invert_yaxis()
    viz_style.save(os.path.join(CHART_DIR, 'eval_public_accuracy.png'))

    average_latency = float(np.mean(latencies))
    _, ax = plt.subplots(figsize=(12, 4.5))
    ax.barh(questions, latencies, color=viz_style.ACCENT, edgecolor=viz_style.BORDER)
    ax.axvline(average_latency, color=viz_style.ACCENT_LIGHT, linestyle='--', linewidth=1.5,
               label=f'Mean: {average_latency:.4f}s')
    ax.set_xlabel('Latency (seconds)')
    ax.set_title('Public Demo Per-Question Latency (deterministic extractive method)')
    ax.legend()
    ax.invert_yaxis()
    viz_style.save(os.path.join(CHART_DIR, 'eval_public_latency.png'))
    print(f'Saved {CHART_DIR}/eval_public_accuracy.png and eval_public_latency.png')


def main():
    results = []
    print(f'{"Question":<52} {"Keyword":<16} {"Hit":<5} {"Time (s)":<8}')
    print('-' * 90)

    for question, keyword in EVAL_QUESTIONS:
        start = time.perf_counter()
        result = demo_runtime.run_turn(question)
        latency = round(time.perf_counter() - start, 4)
        answer = result['final_answer']

        alternatives = KEYWORD_ALTERNATIVES.get(keyword, [keyword])
        hit = any(alt.lower() in answer.lower() for alt in alternatives)

        results.append({
            'question': question, 'answer': answer, 'expected_keyword': keyword,
            'keyword_found': hit, 'latency_seconds': latency,
        })
        print(f'{question:<52} {keyword:<16} {"YES" if hit else "no":<5} {latency:<8}')

    accuracy = sum(r['keyword_found'] for r in results)
    avg_lat = round(sum(r['latency_seconds'] for r in results) / len(results), 4)
    print('-' * 90)
    print(f'\nAccuracy: {accuracy}/{len(results)} ({accuracy * 10:.0f}%)')
    print(f'Average latency: {avg_lat}s per question')

    with open('outputs_demo/public_demo_evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print('Saved outputs_demo/public_demo_evaluation_results.json')

    _plot_charts(results)


if __name__ == '__main__':
    main()
