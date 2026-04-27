import os
import re
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from wordcloud import WordCloud

from preprocessing import clean_for_viz, preprocess_note

def main():
    os.makedirs('static', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    sns.set_theme(style='whitegrid', palette='muted')
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 11

    print('Loading dataset...')
    # Load discharge.csv.gz in chunks of 10,000 rows, but wait, the prompt asks to "loads discharge.csv.gz in chunks of 10,000 rows"
    # Wait, the prompt specifically says "loads discharge.csv.gz in chunks of 10,000 rows"!
    # Let me implement the chunked reading.
    chunks = []
    chunksize = 10000
    for chunk in pd.read_csv('discharge.csv.gz', compression='gzip', chunksize=chunksize):
        chunks.append(chunk)
    df_full = pd.concat(chunks, ignore_index=True)

    print('=== Dataset Overview ===')
    print(f'Total notes:        {len(df_full):,}')
    print(f'Unique patients:    {df_full["subject_id"].nunique():,}')
    print(f'Unique admissions:  {df_full["hadm_id"].nunique():,}')
    print(f'Null text values:   {df_full["text"].isna().sum()}')

    df_full['word_count'] = df_full['text'].dropna().apply(lambda x: len(str(x).split()))
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df_full['word_count'].dropna(), bins=80, color='steelblue', edgecolor='white', linewidth=0.4)
    ax.axvline(df_full['word_count'].mean(), color='tomato', linestyle='--', linewidth=1.5,
               label=f'Mean: {df_full["word_count"].mean():.0f} words')
    ax.axvline(df_full['word_count'].median(), color='orange', linestyle='--', linewidth=1.5,
               label=f'Median: {df_full["word_count"].median():.0f} words')
    ax.set_xlabel('Note length (words)')
    ax.set_ylabel('Number of notes')
    ax.set_title('Distribution of MIMIC-IV Discharge Note Lengths')
    ax.legend()
    plt.tight_layout()
    plt.savefig('static/eda_note_length.png', bbox_inches='tight')
    plt.close()

    df_full['text_lower'] = df_full['text'].str.lower().fillna('')

    df_diabetes    = df_full[df_full['text_lower'].str.contains('diabetes|diabetic|insulin|hba1c|hyperglycemi', regex=True)].copy()
    df_cardiac     = df_full[df_full['text_lower'].str.contains('cardiac|heart failure|myocardial|coronary|atrial fibrillation', regex=True)].copy()
    df_respiratory = df_full[df_full['text_lower'].str.contains('respiratory|copd|pneumonia|asthma|pulmonary', regex=True)].copy()

    conditions = ['Diabetes', 'Cardiac', 'Respiratory']
    sizes = [len(df_diabetes), len(df_cardiac), len(df_respiratory)]
    colors = ['#4C9BE8', '#E87B4C', '#4CE88A']

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(conditions, sizes, color=colors, edgecolor='white', linewidth=0.5)
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{size:,}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Number of notes')
    ax.set_title('Condition Subset Sizes')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.tight_layout()
    plt.savefig('static/eda_condition_subsets.png', bbox_inches='tight')
    plt.close()

    SECTION_HEADERS = [
        'Chief Complaint', 'History of Present Illness', 'Past Medical History',
        'Social History', 'Family History', 'Physical Exam', 'Pertinent Results',
        'Brief Hospital Course', 'Medications on Admission', 'Discharge Medications',
        'Discharge Disposition', 'Discharge Diagnosis', 'Discharge Condition',
        'Discharge Instructions', 'Followup Instructions'
    ]

    sample = df_diabetes.head(2000)
    section_counts = {h: sample['text'].str.contains(h, case=False, na=False).sum() for h in SECTION_HEADERS}

    section_df = pd.DataFrame(list(section_counts.items()), columns=['Section', 'Count'])
    section_df['Pct'] = (section_df['Count'] / len(sample) * 100).round(1)
    section_df = section_df.sort_values('Count', ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(section_df['Section'], section_df['Pct'], color='steelblue', edgecolor='white')
    ax.set_xlabel('% of notes containing section')
    ax.set_title('Section Header Frequency (first 2,000 notes)')
    ax.axvline(90, color='tomato', linestyle='--', linewidth=1, label='90% threshold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('static/eda_section_headers.png', bbox_inches='tight')
    plt.close()

    raw_text = ' '.join(df_diabetes['text'].dropna().head(500).tolist()).lower()
    raw_words = re.findall(r'\b[a-z]{3,}\b', raw_text)
    top_raw = Counter(raw_words).most_common(20)
    words_r, counts_r = zip(*top_raw)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(words_r, counts_r, color='slategray', edgecolor='white')
    ax.set_title('Top 20 Words Before Preprocessing')
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('static/eda_wordfreq_before.png', bbox_inches='tight')
    plt.close()

    print('Generating word clouds...')
    full_sample_cleaned = ' '.join([clean_for_viz(str(t)) for t in df_full['text'].dropna().head(800).tolist()])
    wc_full = WordCloud(width=900, height=420, background_color='white', colormap='Blues', max_words=100, min_word_length=4, collocations=False).generate(full_sample_cleaned)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.imshow(wc_full, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud — Full Dataset', fontsize=12)
    plt.tight_layout()
    plt.savefig('static/eda_wordcloud_full.png', bbox_inches='tight')
    plt.close()

    diabetes_sample_cleaned = ' '.join([clean_for_viz(str(t)) for t in df_diabetes['text'].dropna().head(800).tolist()])
    wc_diabetes = WordCloud(width=900, height=420, background_color='white', colormap='Greens', max_words=100, min_word_length=4, collocations=False).generate(diabetes_sample_cleaned)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.imshow(wc_diabetes, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud — Diabetes Subset', fontsize=12)
    plt.tight_layout()
    plt.savefig('static/eda_wordcloud_diabetes.png', bbox_inches='tight')
    plt.close()

    print('Preprocessing 10,000 diabetes notes...')
    df_diabetes_pp = df_diabetes.head(10000).copy()
    df_diabetes_pp['cleaned_text'] = df_diabetes_pp['text'].apply(preprocess_note)
    df_diabetes_pp = df_diabetes_pp[df_diabetes_pp['cleaned_text'].str.split().str.len() >= 30]
    df_diabetes_pp = df_diabetes_pp.reset_index(drop=True)
    df_diabetes_pp.to_pickle('outputs/df_diabetes_pp.pkl')
    print('Saved preprocessed data to outputs/df_diabetes_pp.pkl')

    cleaned_text_sample = ' '.join(df_diabetes_pp['cleaned_text'].head(500).tolist())
    words_clean = re.findall(r'\b[a-z]{3,}\b', cleaned_text_sample)
    top_clean = Counter(words_clean).most_common(20)
    words_c, counts_c = zip(*top_clean)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(words_c, counts_c, color='steelblue', edgecolor='white')
    ax.set_title('Top 20 Words After Preprocessing')
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('static/eda_wordfreq_after.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
