"""Builds the two cached artifacts scripts/regenerate_ui_charts.py needs
(df_notes_pp.pkl, chunks_temp.pkl) from the synthetic demo notes, mirroring
the schema data.py and chunking.py produce for the real pipeline (see
train.py). Writes into outputs_demo/ only -- never touches outputs/ or reads
any real MIMIC-IV data.

Run once, whenever scripts/synthetic_demo_notes.py changes, before
regenerating the demo's own System Overview charts:
    OUTPUT_DIR=outputs_demo/ python -m scripts.build_demo_chart_inputs
    OUTPUT_DIR=outputs_demo/ python -m scripts.regenerate_ui_charts
"""
import os
import pickle

import pandas as pd

import chunking
import preprocessing
from scripts.synthetic_demo_notes import DEMO_NOTES

DEMO_OUTPUT_DIR = 'outputs_demo'


def main():
    os.makedirs(DEMO_OUTPUT_DIR, exist_ok=True)

    df = pd.DataFrame(DEMO_NOTES)
    df['word_count'] = df['text'].apply(lambda value: len(str(value).split()))
    df['cleaned_text'] = df['text'].apply(preprocessing.preprocess_note)
    df.to_pickle(os.path.join(DEMO_OUTPUT_DIR, 'df_notes_pp.pkl'))
    print(f'Saved {DEMO_OUTPUT_DIR}/df_notes_pp.pkl ({len(df)} synthetic notes)')

    all_chunks, chunk_sources = chunking.chunk_all_notes(df, text_col='text', max_notes=len(df))
    with open(os.path.join(DEMO_OUTPUT_DIR, 'chunks_temp.pkl'), 'wb') as f:
        pickle.dump({'chunks': all_chunks, 'sources': chunk_sources}, f)
    print(f'Saved {DEMO_OUTPUT_DIR}/chunks_temp.pkl ({len(all_chunks)} chunks)')


if __name__ == '__main__':
    main()
