"""Extractive answering: decompose, retrieve per part, select verbatim evidence.

The generative path asks FLAN-T5 to write an answer from the passages. On the
evaluation dev split it cut medication lists short, answered only one half of
two-part questions, and almost never said the notes lacked the answer. This
module answers by selection instead:

1. Decompose a two-part question ("X, and which Y?") into parts, as in
   decomposed prompting (Khot et al., ICLR 2023) and Self-Ask (Press et al.,
   2023). Each part is retrieved separately.
2. Split the retrieved passages into answer units: sentences, except that a
   numbered list (a medication list) stays one unit so it is never cut short.
3. Score every unit against its part with a local cross encoder and keep the
   best one. Returning note text verbatim, as extractive EHR QA does (emrQA,
   Pampari et al., 2018), means no dose or value can be paraphrased wrongly.
4. Refuse when no unit scores above a threshold ("negative rejection", Chen et
   al., AAAI 2024). The threshold is chosen on the dev split only.
"""
import re

import config

_PART_SPLIT = re.compile(
    r',?\s+and\s+(?=(?:(?:over|for|by|at|in)\s+)?(?:which|what|how|when|where|who|did|does|is|was|were)\b)',
    re.IGNORECASE)
# "the heart failure patient", "the 45 year old with new diabetes",
# "the patient with the upper gastrointestinal bleed"
_PATIENT = re.compile(
    r"\bthe\s+(?:(?!(?:the|of|in|on|at|for)\b)[\w']+\s+){0,5}?(?:patient|year old)"
    r"(?:\s+(?:with|who|treated|admitted|newly|diagnosed)\b[^,?]*)?", re.IGNORECASE)
_BARE_PATIENT = re.compile(r'\bthe patient\b(?!\s+(?:with|who|treated|admitted))', re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
_NUMBERED_ITEM = re.compile(r'(?:^|\s)\d{1,2}\.\s+[A-Za-z]')
_HEADER = re.compile(r'^\[[^\]]+\]\s*')

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(config.EXTRACTIVE_RERANKER)
    return _reranker


def decompose(question):
    """Split a two-part question into self-contained parts. A part that does
    not name the patient gets the patient phrase from the other part, so
    "What was the creatinine of the heart failure patient, and which
    supplement was prescribed?" asks both parts about the same patient."""
    parts = [p.strip(' ,?') for p in _PART_SPLIT.split(question) if p.strip(' ,?')]
    if len(parts) < 2:
        return [question]
    def named(part):
        match = _PATIENT.search(part)
        return match.group(0) if match and not _BARE_PATIENT.fullmatch(match.group(0)) else None

    patient = next((p for p in map(named, parts) if p), None)
    out = []
    for part in parts:
        if patient and not named(part):
            if _BARE_PATIENT.search(part):
                part = _BARE_PATIENT.sub(patient, part, count=1)
            else:
                part = f'{part} for {patient}'
        out.append(part + '?')
    return out


def is_list_unit(unit):
    """True for a numbered list unit, such as a medication list."""
    return len(_NUMBERED_ITEM.findall(unit)) >= 2


def answer_units(chunk_text):
    """Sentences of a passage, with a numbered list kept as one unit."""
    body = _HEADER.sub('', chunk_text).strip()
    if is_list_unit(body):
        return [body]
    return [s.strip() for s in _SENTENCE_SPLIT.split(body) if len(s.split()) >= 3]


def information_need(question):
    """The question without its patient description. Retrieval has already
    used the description to find the right note; left in, it makes the
    selector prefer a sentence that repeats the description over the one
    that answers the question."""
    need = _PATIENT.sub('', question)
    need = re.sub(r'\b(?:for|of|in|did|does|was|is)\s*(?=[?,]|$)', '', need.strip())
    return re.sub(r'\s+', ' ', need).strip(' ,') or question


def best_unit(question, chunks, reranker=None, top_chunks=None):
    """(score, unit, chunk) for the highest scoring unit in the first
    top_chunks retrieved chunks (default config.EXTRACTIVE_TOP_CHUNKS), or None."""
    top_chunks = config.EXTRACTIVE_TOP_CHUNKS if top_chunks is None else top_chunks
    candidates = [(unit, chunk) for chunk in chunks[:top_chunks] for unit in answer_units(chunk['chunk_text'])]
    if not candidates:
        return None
    need = information_need(question)
    scores = (reranker or get_reranker()).predict([(need, unit) for unit, _ in candidates])
    best = max(range(len(candidates)), key=lambda i: float(scores[i]))
    return float(scores[best]), candidates[best][0], candidates[best][1]


def extractive_answer(parts_with_chunks, threshold=None, reranker=None):
    """parts_with_chunks: [(part question, retrieved chunks)]. Returns the
    joined verbatim units, or None when no part clears the threshold."""
    threshold = config.EXTRACTIVE_THRESHOLD if threshold is None else threshold
    units = []
    for part, chunks in parts_with_chunks:
        found = best_unit(part, chunks, reranker)
        if found and found[0] >= threshold and found[1] not in units:
            units.append(found[1])
    if not units:
        return None
    return ' '.join(u if u.endswith(('.', '!', '?')) else u + '.' for u in units)


def hybrid_answer(part_evidence, generator, judge=None, reranker=None, threshold=None):
    """Answer from [(part question, retrieved chunks)]. Returns (answer, refused).
    A part whose best unit is a numbered list is answered with the list
    verbatim; any other part is answered by the generator from its passages.
    The whole question is refused only when the judge says no to every part
    and no part's best unit reaches the threshold."""
    from core.generation import generate_answer, get_judge, is_answerable
    threshold = config.EXTRACTIVE_THRESHOLD if threshold is None else threshold
    judge = judge or get_judge()
    texts, judged_answerable, best_scores = [], [], []
    for part, chunks in part_evidence:
        if not chunks:
            continue
        found = best_unit(part, chunks, reranker)
        best_scores.append(found[0] if found else float('-inf'))
        judged_answerable.append(is_answerable(part, chunks, judge))
        if found and is_list_unit(found[1]):
            texts.append(found[1] if found[1].endswith('.') else found[1] + '.')
        else:
            texts.append(generate_answer(part, chunks[:config.GENERATION_TOP_K], generator)[0])
    if not texts or (not any(judged_answerable) and max(best_scores) < threshold):
        return None, True
    return ' '.join(dict.fromkeys(texts)), False
