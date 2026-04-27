import time
import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

def main():
    with open('outputs/chunks_temp.pkl', 'rb') as f:
        data = pickle.load(f)
    all_chunks = data['chunks']
    chunk_sources = data['sources']

    print('Loading all-MiniLM-L6-v2...')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print('Model loaded.')

    batch_size = 64
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
        pickle.dump({'chunks': all_chunks, 'sources': chunk_sources}, f)

    print('Outputs saved: outputs/faiss_index.index and outputs/chunks_data.pkl')

if __name__ == '__main__':
    main()
