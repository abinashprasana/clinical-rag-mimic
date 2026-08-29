"""Builds a FAISS index + chunk store from the fully synthetic demo notes in
scripts/synthetic_demo_notes.py, writing to outputs_demo/ -- a separate,
git-safe directory from outputs/ (which holds the real MIMIC-IV-derived
index and must never be committed).

Reuses the same chunking (chunking.section_chunk) and embedding/index logic
(embeddings.py) the real pipeline uses, so the demo index is retrieved and
generated against exactly the same way -- just pointed at synthetic data.

Run once, whenever scripts/synthetic_demo_notes.py changes:
    python scripts/build_demo_index.py

To actually serve from this index instead of the real one, set in .env:
    OUTPUT_DIR=outputs_demo/
    DATASET_LABEL=Synthetic demo (fabricated notes; no patient data; real MIMIC-IV-Note data requires credentialed PhysioNet access)
"""
import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import chunking
import config
from scripts.synthetic_demo_notes import DEMO_NOTES

DEMO_OUTPUT_DIR = 'outputs_demo'


def main():
    os.makedirs(DEMO_OUTPUT_DIR, exist_ok=True)

    print(f'Chunking {len(DEMO_NOTES)} synthetic notes...')
    all_chunks, provenance = [], []
    for note in DEMO_NOTES:
        note_chunks = chunking.section_chunk(note['text'])
        all_chunks.extend(note_chunks)
        provenance.extend([{'subject_id': note['subject_id'], 'hadm_id': note['hadm_id']}] * len(note_chunks))

    print(f'Total chunks: {len(all_chunks)}')

    print(f'Loading {config.EMBEDDING_MODEL}...')
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    embeddings = model.encode(all_chunks, show_progress_bar=False).astype('float32')
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f'FAISS index built: {index.ntotal} vectors, dimension {embeddings.shape[1]}.')

    faiss.write_index(index, os.path.join(DEMO_OUTPUT_DIR, 'faiss_index.index'))
    with open(os.path.join(DEMO_OUTPUT_DIR, 'chunks_data.pkl'), 'wb') as f:
        pickle.dump({'chunks': all_chunks, 'provenance': provenance}, f)

    print(f'Saved {DEMO_OUTPUT_DIR}/faiss_index.index and {DEMO_OUTPUT_DIR}/chunks_data.pkl')


if __name__ == '__main__':
    main()
