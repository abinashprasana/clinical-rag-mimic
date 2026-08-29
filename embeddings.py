import time
import os
import faiss
import numpy as np
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
import config

def _safe_int(value):
    return int(value) if pd.notna(value) else None

def build_provenance(chunk_sources, df_notes):
    """Map each chunk's originating dataframe row index to its subject_id/hadm_id,
    so retrieved chunks can be cited back to a real patient admission."""
    return [
        {
            'subject_id': _safe_int(df_notes.loc[src, 'subject_id']),
            'hadm_id': _safe_int(df_notes.loc[src, 'hadm_id']),
        }
        for src in chunk_sources
    ]

def main():
    with open('outputs/chunks_temp.pkl', 'rb') as f:
        data = pickle.load(f)
    all_chunks = data['chunks']
    chunk_sources = data['sources']

    df_notes = pd.read_pickle('outputs/df_notes_pp.pkl')
    provenance = build_provenance(chunk_sources, df_notes)

    print(f'Loading {config.EMBEDDING_MODEL}...')
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    print('Model loaded.')

    batch_size = config.EMBEDDING_BATCH_SIZE
    all_embeddings = []
    start_time = time.time()

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        all_embeddings.append(model.encode(batch, show_progress_bar=False))
        if (i // batch_size) % 20 == 0:
            print(f'  {min(i + batch_size, len(all_chunks)):,} / {len(all_chunks):,} chunks embedded...')

    embeddings = np.vstack(all_embeddings).astype('float32')
    print(f'Done. Shape: {embeddings.shape}  |  Time: {time.time() - start_time:.1f}s')

    # FAISS FIX: Normalise embeddings to use Inner Product (Cosine Similarity)
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print('FAISS index built.')
    print(f'  Type:            IndexFlatIP (Cosine Similarity)')
    print(f'  Dimension:       {dimension}')
    print(f'  Vectors stored:  {index.ntotal:,}')

    os.makedirs('outputs', exist_ok=True)
    faiss.write_index(index, 'outputs/faiss_index.index')

    with open('outputs/chunks_data.pkl', 'wb') as f:
        pickle.dump(
            {'chunks': all_chunks, 'sources': chunk_sources, 'provenance': provenance}, f
        )

    print('Outputs saved: outputs/faiss_index.index and outputs/chunks_data.pkl')

if __name__ == '__main__':
    main()
