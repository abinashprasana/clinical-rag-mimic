"""Regenerate UI charts from existing cached outputs without rerunning models.

This script reads the current preprocessed note sample, cached chunks, and
evaluation JSON. It writes PNG files only beneath the configured, ignored
``OUTPUT_DIR`` and does not mutate the retrieval index, cached dataframes,
chunks, or evaluation results. Generated charts must never be copied into the
public ``static`` directory when they come from restricted data.
"""

from collections import Counter
import json
import os
import pickle
import re

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from wordcloud import WordCloud

import config
import core.viz_style as viz_style


OUTPUT_DIR = config.OUTPUT_DIR
CHART_DIR = os.path.join(OUTPUT_DIR, 'ui_charts')
WORD_COLORS = [
    viz_style.ACCENT_LIGHT,
    viz_style.ACCENT,
    '#70B8C2',
    '#5897A4',
    '#A8BBC1',
]


def _word_color(*_args, random_state=None, **_kwargs):
    return random_state.choice(WORD_COLORS)


def _save(name):
    viz_style.save(os.path.join(CHART_DIR, name))


def plot_note_lengths(df_notes):
    word_counts = df_notes['word_count'].dropna()
    if word_counts.empty:
        word_counts = df_notes['text'].dropna().map(lambda value: len(str(value).split()))

    mean_len = word_counts.mean()
    median_len = word_counts.median()
    p99 = word_counts.quantile(0.99)
    hidden = int((word_counts > p99).sum())

    _, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(
        word_counts[word_counts <= p99],
        bins=64,
        color=viz_style.ACCENT,
        edgecolor=viz_style.BORDER,
        linewidth=0.4,
    )
    ax.axvline(mean_len, color=viz_style.PALETTE[3], linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_len:.0f} words')
    ax.axvline(median_len, color=viz_style.ACCENT_LIGHT, linestyle='--', linewidth=1.5,
               label=f'Median: {median_len:.0f} words')
    ax.set_xlabel('Note length (words)')
    ax.set_ylabel('Number of notes')
    ax.set_title(f'Indexed Sample Note Lengths (n={len(word_counts):,})')
    ax.text(
        0.98,
        0.95,
        f'{hidden:,} notes above the 99th percentile are not shown',
        transform=ax.transAxes,
        ha='right',
        va='top',
        color=viz_style.MUTED,
        fontsize=9,
    )
    ax.legend()
    _save('eda_note_length.png')


def plot_condition_coverage(df_notes):
    text = df_notes['text'].fillna('').str.lower()
    definitions = {
        'Diabetes': r'diabetes|diabetic|insulin|hba1c|hyperglycemi',
        'Cardiac': r'cardiac|heart failure|myocardial|coronary|atrial fibrillation',
        'Respiratory': r'respiratory|copd|pneumonia|asthma|pulmonary',
    }
    sizes = [int(text.str.contains(pattern, regex=True).sum()) for pattern in definitions.values()]
    percentages = [size / len(df_notes) * 100 for size in sizes]

    _, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        list(definitions),
        sizes,
        color=viz_style.PALETTE[:3],
        edgecolor=viz_style.BORDER,
        linewidth=0.5,
        width=0.55,
    )
    ceiling = max(sizes) if sizes else 1
    for bar, size, percentage in zip(bars, sizes, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ceiling * 0.02,
            f'{size:,}\n({percentage:.0f}% of sample)',
            ha='center',
            va='bottom',
            fontsize=9,
        )
    ax.set_ylabel('Number of notes')
    ax.set_ylim(0, ceiling * 1.22)
    ax.set_title('Condition Keyword Coverage in the Indexed Sample')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f'{int(value):,}'))
    _save('eda_condition_subsets.png')


def plot_section_headers(df_notes):
    section_sample = df_notes.head(min(2000, len(df_notes)))
    headers = [header for header in config.MIMIC_SECTIONS if header != 'Allergies']
    counts = {
        header: int(section_sample['text'].str.contains(header, case=False, na=False).sum())
        for header in headers
    }
    section_df = pd.DataFrame(list(counts.items()), columns=['Section', 'Count'])
    section_df['Pct'] = section_df['Count'] / len(section_sample) * 100
    section_df = section_df.sort_values('Pct')

    _, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        section_df['Section'],
        section_df['Pct'],
        color=viz_style.ACCENT,
        edgecolor=viz_style.BORDER,
    )
    ax.axvline(90, color=viz_style.ACCENT_LIGHT, linestyle='--', linewidth=1,
               label='90% reference')
    ax.set_xlabel('% of notes containing section')
    ax.set_title(f'Section Header Frequency (n={len(section_sample):,})')
    ax.legend()
    _save('eda_section_headers.png')


def _plot_frequency(text, filename, title, color):
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    common = Counter(words).most_common(20)
    labels, counts = zip(*common)

    _, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, counts, color=color, edgecolor=viz_style.BORDER)
    ax.set_title(title)
    ax.set_ylabel('Frequency')
    ax.tick_params(axis='x', rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')
    _save(filename)


def plot_word_frequencies(df_notes):
    source_text = ' '.join(df_notes['text'].dropna().head(500).astype(str))
    cleaned_text = ' '.join(df_notes['cleaned_text'].dropna().head(500).astype(str))
    _plot_frequency(
        source_text,
        'eda_wordfreq_before.png',
        'Top Terms Before Analysis Preprocessing',
        viz_style.PALETTE[3],
    )
    _plot_frequency(
        cleaned_text,
        'eda_wordfreq_after.png',
        'Top Terms After Analysis Preprocessing',
        viz_style.ACCENT,
    )


def _plot_wordcloud(text, filename, title, seed):
    cloud = WordCloud(
        width=900,
        height=420,
        background_color=viz_style.BG,
        color_func=_word_color,
        max_words=100,
        min_word_length=4,
        collocations=False,
        random_state=seed,
    ).generate(text)
    _, ax = plt.subplots(figsize=(13, 5))
    ax.imshow(cloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    _save(filename)


def plot_wordclouds(df_notes):
    # Both views use only the already-selected local sample. Nothing reaches
    # beyond OUTPUT_DIR or reads a second slice from the credentialed source.
    broad_text = ' '.join(df_notes['text'].dropna().head(800).astype(str))
    indexed_text = ' '.join(df_notes['cleaned_text'].dropna().head(800).astype(str))
    _plot_wordcloud(broad_text, 'eda_wordcloud_full.png', 'Indexed Raw-Text Vocabulary', config.RANDOM_SEED)
    _plot_wordcloud(indexed_text, 'eda_wordcloud_sample.png', 'Indexed Sample Vocabulary', config.RANDOM_SEED)


def plot_chunk_diagnostics():
    with open(os.path.join(OUTPUT_DIR, 'chunks_temp.pkl'), 'rb') as handle:
        cached = pickle.load(handle)
    all_chunks = cached['chunks']
    chunk_sources = cached['sources']
    chunk_lengths = [len(chunk.split()) for chunk in all_chunks]
    safe_word_limit = 230
    at_risk = sum(length > safe_word_limit for length in chunk_lengths)

    _, ax = plt.subplots(figsize=(10, 4))
    ax.hist(chunk_lengths, bins=60, color=viz_style.ACCENT,
            edgecolor=viz_style.BORDER, linewidth=0.4)
    ax.axvline(config.MAX_CHUNK_WORDS, color=viz_style.ACCENT_LIGHT, linestyle=':', linewidth=1.5,
               label=f'{config.MAX_CHUNK_WORDS}-word content split target')
    ax.axvline(safe_word_limit, color=viz_style.PALETTE[3], linestyle='--', linewidth=1.5,
               label=f'~{safe_word_limit}-word embedding safety reference')
    ax.set_xlabel('Chunk length (words, including section prefix)')
    ax.set_ylabel('Number of chunks')
    ax.set_title(f'Chunk Lengths ({at_risk}/{len(chunk_lengths):,} above safety reference)')
    ax.legend()
    _save('eda_chunk_lengths.png')

    chunks_per_note = Counter(chunk_sources)
    counts = list(chunks_per_note.values())
    _, ax = plt.subplots(figsize=(10, 4))
    ax.hist(counts, bins=range(1, max(counts) + 2), color=viz_style.PALETTE[3],
            edgecolor=viz_style.BORDER, align='left')
    ax.axvline(np.mean(counts), color=viz_style.ACCENT_LIGHT, linestyle='--', linewidth=1.5,
               label=f'Mean: {np.mean(counts):.1f} chunks/note')
    ax.set_xlabel('Chunks per note')
    ax.set_ylabel('Number of notes')
    ax.set_title('Chunks Produced per Note')
    ax.legend()
    _save('eda_chunks_per_note.png')


def plot_evaluation():
    with open(os.path.join(OUTPUT_DIR, 'evaluation_results.json'), encoding='utf-8') as handle:
        results = json.load(handle)

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
    ax.set_title(f'Keyword-Hit Evaluation ({accuracy}/{len(results)} questions)')
    ax.legend(
        handles=[
            mpatches.Patch(color=viz_style.SUCCESS, label='Keyword found'),
            mpatches.Patch(color=viz_style.DANGER, label='Keyword not found'),
        ],
        loc='lower right',
    )
    ax.invert_yaxis()
    _save('eval_accuracy.png')

    average_latency = float(np.mean(latencies))
    _, ax = plt.subplots(figsize=(12, 4.5))
    ax.barh(questions, latencies, color=viz_style.ACCENT, edgecolor=viz_style.BORDER)
    ax.axvline(average_latency, color=viz_style.ACCENT_LIGHT, linestyle='--', linewidth=1.5,
               label=f'Mean: {average_latency:.2f}s')
    ax.set_xlabel('Latency (seconds)')
    ax.set_title('Recorded Per-Question Latency')
    ax.legend()
    ax.invert_yaxis()
    _save('eval_latency.png')


def main():
    os.makedirs(CHART_DIR, exist_ok=True)
    viz_style.apply_style()
    notes = pd.read_pickle(os.path.join(OUTPUT_DIR, 'df_notes_pp.pkl'))

    plot_note_lengths(notes)
    plot_condition_coverage(notes)
    plot_section_headers(notes)
    plot_word_frequencies(notes)
    plot_wordclouds(notes)
    plot_chunk_diagnostics()
    plot_evaluation()
    print(f'Regenerated 11 local charts under {CHART_DIR}.')


if __name__ == '__main__':
    main()
