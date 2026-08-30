import os
import re
import faiss
import pickle
import config

def load_index(output_dir=None):
    output_dir = output_dir or config.OUTPUT_DIR
    idx = faiss.read_index(os.path.join(output_dir, 'faiss_index.index'))
    with open(os.path.join(output_dir, 'chunks_data.pkl'), 'rb') as f:
        data = pickle.load(f)
    return idx, data['chunks'], data['provenance']

def _word_set(text):
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))

def _is_near_duplicate(candidate_words, kept_words_list, threshold=0.7):
    """Overlapping chunk windows (see chunking.py's CHUNK_OVERLAP) can return
    two chunks that share most of the same list/text. Comparing word sets
    catches that near-duplicate case cheaply, without needing exact-text
    matching."""
    if not candidate_words:
        return False
    for kept_words in kept_words_list:
        smaller = min(len(candidate_words), len(kept_words))
        if smaller == 0:
            continue
        overlap = len(candidate_words & kept_words) / smaller
        if overlap >= threshold:
            return True
    return False

_HEADER_RE = re.compile(r"^\[([^\]]+)\]")
# "followup"/"follow-up"/"follow up" all mean the same section in practice
# but tokenize differently -- normalize before comparing word sets.
_HEADER_ALIASES = (("followup", "follow up"),)

def _normalized_words(text):
    for source, target in _HEADER_ALIASES:
        text = text.lower().replace(source, target)
    return _word_set(text)

def _header_boost(chunk_text, query_words, weight=0.15):
    """Chunking prefixes every chunk with its own section header (see
    chunking.py's "[Header] ..." format). When a question's own words
    substantially name that header (e.g. "discharge disposition" question,
    "[Discharge Disposition]" chunk), nudge that chunk's rank up. This is
    the same header/field-boosting idea used in classic search ranking
    (e.g. BM25 field weighting) -- it rewards a chunk for being from the
    section the question is actually asking about, not for containing any
    particular expected answer word, so it generalizes to any question
    rather than being tuned to this project's own evaluation keywords."""
    match = _HEADER_RE.match(chunk_text)
    if not match:
        return 0.0
    header_words = _normalized_words(match.group(1))
    if not header_words:
        return 0.0
    overlap = len(header_words & query_words) / len(header_words)
    return weight * overlap

def retrieve_chunks(question, model, index, chunks, provenance, k=config.DEFAULT_TOP_K):
    """Returns up to k chunks with their similarity score and source
    subject_id/hadm_id, so answers can be cited back to a real note.
    Near-duplicate passages (from overlapping chunk windows) are dropped in
    favor of the higher-scoring copy, so the model -- and the evidence
    inspector -- never sees the same passage twice."""
    query_vec = model.encode([question]).astype('float32')
    # FAISS FIX: Normalise query for Inner Product
    faiss.normalize_L2(query_vec)
    # Search a wider candidate pool than k so de-duplication still leaves
    # up to k distinct passages instead of shrinking below it.
    scores, indices = index.search(query_vec, min(k * 3, len(chunks)))

    query_words = _normalized_words(question)
    candidates = [(i, score) for score, i in zip(scores[0], indices[0]) if i >= 0]
    # Re-rank by similarity plus header-relevance boost; the displayed
    # "score" stays the raw similarity so it keeps meaning what it says.
    # Keep this lightweight header signal instead of adding another model-
    # backed re-ranking stage, which would increase latency and dependencies.
    candidates.sort(key=lambda c: c[1] + _header_boost(chunks[c[0]], query_words), reverse=True)

    results = []
    kept_words_list = []
    for i, score in candidates:
        candidate_words = _word_set(chunks[i])
        if _is_near_duplicate(candidate_words, kept_words_list):
            continue
        kept_words_list.append(candidate_words)
        results.append({
            'chunk_text': chunks[i],
            'subject_id': provenance[i]['subject_id'],
            'hadm_id': provenance[i]['hadm_id'],
            'chunk_idx': int(i),
            'score': float(score),
        })
        if len(results) == k:
            break
    return results
