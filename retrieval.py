import os
import faiss
import pickle

def load_index(output_dir='outputs/'):
    idx = faiss.read_index(os.path.join(output_dir, 'faiss_index.index'))
    with open(os.path.join(output_dir, 'chunks_data.pkl'), 'rb') as f:
        data = pickle.load(f)
    return idx, data['chunks'], data['sources']

def retrieve_chunks(question, model, index, chunks, k=5):
    query_vec = model.encode([question]).astype('float32')
    # FAISS FIX: Normalise query for Inner Product
    faiss.normalize_L2(query_vec)
    _, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]
