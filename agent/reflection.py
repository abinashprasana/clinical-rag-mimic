"""Local faithfulness check: does the local model's draft answer only make
claims actually supported by the retrieved chunks? Fully on-device, never
calls Gemini -- this is the real safety gate (the optional external
reflect_structure_node in graph.py is advisory only, see its docstring).

Uses a lightweight content-word + numeric-value overlap heuristic rather
than an additional NLI model: it's deterministic, fast enough to run on
every turn (and on every retry) on CPU-only hardware, and numeric mismatches
-- a fabricated dose or lab value -- are exactly the highest-stakes failure
mode in a clinical answer, which this catches directly rather than relying
on an NLI model's fuzzier entailment judgment.
"""
import re

_STOPWORDS = {
    'the', 'a', 'an', 'is', 'was', 'were', 'of', 'for', 'and', 'to', 'in',
    'on', 'at', 'with', 'that', 'this', 'these', 'those', 'it', 'as', 'by',
    'from', 'be', 'are', 'or', 'no', 'not',
}

REFUSAL_PHRASES = ('cannot find this information',)

MIN_OVERLAP_RATIO = 0.4


def _chunk_text(chunk):
    return chunk['chunk_text'] if isinstance(chunk, dict) else chunk


def _content_words(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def _split_claims(answer):
    parts = re.split(r'(?<=[.!?])\s+', answer.strip())
    return [p.strip() for p in parts if p.strip()]


def local_reflect(draft_answer, retrieved_chunks):
    """Returns {'supported': bool, 'unsupported_claims': [...]}."""
    if not draft_answer:
        return {'supported': True, 'unsupported_claims': []}

    if any(p in draft_answer.lower() for p in REFUSAL_PHRASES):
        return {'supported': True, 'unsupported_claims': []}

    if not retrieved_chunks:
        # nothing to check the answer against -- a direct-response turn
        # with no retrieval isn't a faithfulness question.
        return {'supported': True, 'unsupported_claims': []}

    context_text = ' '.join(_chunk_text(c) for c in retrieved_chunks)
    context_words = _content_words(context_text)
    context_numbers = _numbers(context_text)

    unsupported = []
    for claim in _split_claims(draft_answer):
        claim_words = _content_words(claim)
        if not claim_words:
            continue
        overlap_ratio = len(claim_words & context_words) / len(claim_words)
        claim_numbers = _numbers(claim)
        numbers_ok = claim_numbers <= context_numbers
        if overlap_ratio < MIN_OVERLAP_RATIO or not numbers_ok:
            unsupported.append(claim)

    return {'supported': len(unsupported) == 0, 'unsupported_claims': unsupported}
