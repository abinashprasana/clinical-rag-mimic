"""Precomputes Gemini embeddings for the synthetic demo chunks, so the public
Vercel deployment (demo/runtime.py) can do real retrieval via a lightweight
API call instead of loading sentence-transformers locally.

Reuses the exact same chunks outputs_demo/chunks_data.pkl holds (produced by
demo/build_index.py via chunking.section_chunk on the fabricated
notes), so the public deployment retrieves against the identical corpus the
local full pipeline does -- only the embedding model differs (Gemini's
hosted API instead of a locally-loaded sentence-transformer).

Requires GEMINI_API_KEY to be set (in .env locally). Writes a small file
(~101 chunks x 3072 floats, well under 2MB) that is safe and necessary to
commit: it contains only vectors and the fabricated chunk text/provenance
already public in demo/notes.py, nothing from restricted
data.

Run once, whenever demo/notes.py changes:
    python -m demo.build_gemini_embeddings
"""
import pickle
import time

import numpy as np
from google import genai
from google.genai.errors import ClientError

import config

EMBEDDING_MODEL = 'models/gemini-embedding-001'
CHUNKS_PATH = 'outputs_demo/chunks_data.pkl'
OUTPUT_PATH = 'outputs_demo/gemini_chunk_embeddings.pkl'
# Free tier caps embed_content at 100 requests/minute, counted per chunk in
# the batch, not per API call -- so a batch this size leaves headroom before
# the next batch instead of bursting the whole quota in one call.
BATCH_SIZE = 20


def _embed_with_retry(client, batch, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            return client.models.embed_content(model=EMBEDDING_MODEL, contents=batch)
        except ClientError as exc:
            if exc.code != 429 or attempt == max_attempts - 1:
                raise
            print(f'  rate limited, waiting 60s (attempt {attempt + 1}/{max_attempts})...')
            time.sleep(60)


def main():
    if not config.GEMINI_API_KEY:
        raise SystemExit('GEMINI_API_KEY is not set. Add it to .env before running this script.')

    with open(CHUNKS_PATH, 'rb') as f:
        data = pickle.load(f)
    chunks, provenance = data['chunks'], data['provenance']
    print(f'Embedding {len(chunks)} synthetic chunks via {EMBEDDING_MODEL}...')

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    vectors = []
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        resp = _embed_with_retry(client, batch)
        vectors.extend(e.values for e in resp.embeddings)
        print(f'  embedded {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}')
        if start + BATCH_SIZE < len(chunks):
            time.sleep(15)

    embeddings = np.array(vectors, dtype='float32')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-9, None)

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump({
            'chunks': chunks,
            'provenance': provenance,
            'embeddings': embeddings,
            'model': EMBEDDING_MODEL,
        }, f)
    print(f'Saved {OUTPUT_PATH} ({embeddings.shape[0]} vectors, dim {embeddings.shape[1]}).')


if __name__ == '__main__':
    main()
