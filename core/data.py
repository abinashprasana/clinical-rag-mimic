import os
import re
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from wordcloud import WordCloud

from core.preprocessing import clean_for_viz, preprocess_note
import core.viz_style as viz_style
import config

DATA_PATH = config.DATA_PATH

# Random sample size drawn from the FULL dataset (not condition-filtered) that
# gets preprocessed, chunked and embedded into the RAG knowledge base. Kept in
# the low thousands so embedding still finishes in a reasonable time on CPU.
SAMPLE_SIZE = config.SAMPLE_SIZE
RANDOM_SEED = config.RANDOM_SEED

def main():
    os.makedirs('static', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    viz_style.apply_style()

    print('Loading dataset...')
    chunks = []
    chunksize = 10000
    for chunk in pd.read_csv(DATA_PATH, compression='gzip', chunksize=chunksize):
        chunks.append(chunk)
    df_full = pd.concat(chunks, ignore_index=True)

    print('=== Dataset Overview ===')
    print(f'Total notes:        {len(df_full):,}')
    print(f'Unique patients:    {df_full["subject_id"].nunique():,}')
    print(f'Unique admissions:  {df_full["hadm_id"].nunique():,}')
    print(f'Null text values:   {df_full["text"].isna().sum()}')

    word_counts = df_full['word_count'] = df_full['text'].dropna().apply(lambda x: len(str(x).split()))
    mean_len, median_len = word_counts.mean(), word_counts.median()

    # A handful of extreme outlier notes stretch the x-axis so far that the
    # bulk of the distribution gets compressed into a sliver on the left.
    # Clip the view to the 99th percentile and call out what's hidden, rather
    # than letting a few outliers dominate the plot.
    p99 = word_counts.quantile(0.99)
    hidden = int((word_counts > p99).sum())

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(word_counts[word_counts <= p99], bins=80, color=viz_style.PALETTE[0],
             edgecolor=viz_style.BORDER, linewidth=0.4)
    ax.axvline(mean_len, color=viz_style.PALETTE[3], linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_len:.0f} words')
    ax.axvline(median_len, color=viz_style.PALETTE[1], linestyle='--', linewidth=1.5,
               label=f'Median: {median_len:.0f} words')
    ax.set_xlabel('Note length (words)')
    ax.set_ylabel('Number of notes')
    ax.set_title(f'Distribution of MIMIC-IV Discharge Note Lengths (n={len(word_counts):,})')
    ax.text(0.98, 0.95, f'{hidden:,} notes longer than {p99:,.0f} words\nnot shown (99th percentile cutoff)',
            transform=ax.transAxes, ha='right', va='top', fontsize=9, color=viz_style.MUTED)
    ax.legend()
    viz_style.save('static/eda_note_length.png')

    df_full['text_lower'] = df_full['text'].str.lower().fillna('')

    df_diabetes    = df_full[df_full['text_lower'].str.contains('diabetes|diabetic|insulin|hba1c|hyperglycemi', regex=True)].copy()
    df_cardiac     = df_full[df_full['text_lower'].str.contains('cardiac|heart failure|myocardial|coronary|atrial fibrillation', regex=True)].copy()
    df_respiratory = df_full[df_full['text_lower'].str.contains('respiratory|copd|pneumonia|asthma|pulmonary', regex=True)].copy()

    conditions = ['Diabetes', 'Cardiac', 'Respiratory']
    sizes = [len(df_diabetes), len(df_cardiac), len(df_respiratory)]
    pct_of_total = [s / len(df_full) * 100 for s in sizes]
    colors = viz_style.PALETTE[:3]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(conditions, sizes, color=colors, edgecolor=viz_style.BORDER, linewidth=0.5, width=0.55)
    for bar, size, pct in zip(bars, sizes, pct_of_total):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(sizes) * 0.015,
                f'{size:,}\n({pct:.0f}% of dataset)', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Number of notes')
    ax.set_ylim(0, max(sizes) * 1.2)
    ax.set_title('Condition Subset Sizes (keyword-matched, not mutually exclusive)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    viz_style.save('static/eda_condition_subsets.png')

    SECTION_HEADERS = [h for h in config.MIMIC_SECTIONS if h != 'Allergies']

    # A random sample drawn from the FULL dataset (across all conditions) is
    # what actually gets preprocessed, chunked and embedded into the RAG
    # knowledge base, so the EDA below is computed on that same sample rather
    # than on a single-condition subset.
    df_sample = df_full[df_full['text'].notna()].sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).copy()

    section_sample = df_sample.head(2000)
    section_counts = {h: section_sample['text'].str.contains(h, case=False, na=False).sum() for h in SECTION_HEADERS}

    section_df = pd.DataFrame(list(section_counts.items()), columns=['Section', 'Count'])
    section_df['Pct'] = (section_df['Count'] / len(section_sample) * 100).round(1)
    section_df = section_df.sort_values('Count', ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(section_df['Section'], section_df['Pct'], color=viz_style.PALETTE[0], edgecolor=viz_style.BORDER)
    ax.set_xlabel('% of notes containing section')
    ax.set_title('Section Header Frequency (random sample, n=2,000)')
    ax.axvline(90, color=viz_style.PALETTE[3], linestyle='--', linewidth=1, label='90% threshold')
    ax.legend()
    viz_style.save('static/eda_section_headers.png')

    raw_text = ' '.join(df_sample['text'].dropna().head(500).tolist()).lower()
    raw_words = re.findall(r'\b[a-z]{3,}\b', raw_text)
    top_raw = Counter(raw_words).most_common(20)
    words_r, counts_r = zip(*top_raw)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(words_r, counts_r, color=viz_style.PALETTE[3], edgecolor=viz_style.BORDER)
    ax.set_title('Top 20 Words Before Preprocessing')
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    viz_style.save('static/eda_wordfreq_before.png')

    print('Generating word clouds...')
    full_sample_cleaned = ' '.join([clean_for_viz(str(t)) for t in df_full['text'].dropna().head(800).tolist()])
    wc_full = WordCloud(width=900, height=420, background_color=viz_style.BG, colormap='GnBu', max_words=100, min_word_length=4, collocations=False, random_state=config.RANDOM_SEED).generate(full_sample_cleaned)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.imshow(wc_full, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud — Full Dataset', fontsize=12)
    viz_style.save('static/eda_wordcloud_full.png')

    rag_sample_cleaned = ' '.join([clean_for_viz(str(t)) for t in df_sample['text'].dropna().head(800).tolist()])
    wc_sample = WordCloud(width=900, height=420, background_color=viz_style.BG, colormap='GnBu', max_words=100, min_word_length=4, collocations=False, random_state=config.RANDOM_SEED).generate(rag_sample_cleaned)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.imshow(wc_sample, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud — RAG Knowledge Base Sample', fontsize=12)
    viz_style.save('static/eda_wordcloud_sample.png')

    print(f'Preprocessing {SAMPLE_SIZE:,} notes randomly sampled from the full dataset...')
    df_sample['cleaned_text'] = df_sample['text'].apply(preprocess_note)
    df_sample = df_sample[df_sample['cleaned_text'].str.split().str.len() >= 30]
    df_sample = df_sample.reset_index(drop=True)
    df_sample.to_pickle('outputs/df_notes_pp.pkl')
    print('Saved preprocessed data to outputs/df_notes_pp.pkl')

    cleaned_text_sample = ' '.join(df_sample['cleaned_text'].head(500).tolist())
    words_clean = re.findall(r'\b[a-z]{3,}\b', cleaned_text_sample)
    top_clean = Counter(words_clean).most_common(20)
    words_c, counts_c = zip(*top_clean)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(words_c, counts_c, color=viz_style.PALETTE[0], edgecolor=viz_style.BORDER)
    ax.set_title('Top 20 Words After Preprocessing')
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    viz_style.save('static/eda_wordfreq_after.png')

if __name__ == '__main__':
    main()
