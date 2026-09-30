"""Local faithfulness check: does the local model's draft answer only make
claims actually supported by the retrieved chunks? Fully on-device, never
calls Gemini -- this is the real safety gate (the optional external
reflect_structure_node in graph.py is advisory only, see its docstring).
The public runtime (demo/runtime.py) runs this same check on Gemini drafts.

Uses a lightweight content-word + numeric-value overlap heuristic rather
than an additional NLI model: it's deterministic, fast enough to run on
every turn (and on every retry) on CPU-only hardware, and numeric mismatches
-- a fabricated dose or lab value -- are exactly the highest-stakes failure
mode in a clinical answer, which this catches directly rather than relying
on an NLI model's fuzzier entailment judgment.

Three rules on top of word overlap, each added after a synthetic stress test
(evals/stress.py) showed the gap:

* A number must appear in the passages, and when the other words of its
  clause also appear in the passages, the number must appear near at least
  one of them.
  "Donepezil 25 mg" fails when the passages say "Donepezil 10 mg", even if
  25 appears elsewhere in them.
* A claim may not assert a finding the passages mention only as absent:
  "chest pain" fails when the passages only say "no chest pain" (a NegEx
  style scope, Chapman et al. 2001).
* A claim with no content words, such as a lone "31.", still has its
  numbers checked.
"""
import re

_STOPWORDS = {
    'the', 'a', 'an', 'is', 'was', 'were', 'of', 'for', 'and', 'to', 'in',
    'on', 'at', 'with', 'that', 'this', 'these', 'those', 'it', 'as', 'by',
    'from', 'be', 'are', 'or', 'no', 'not',
}

REFUSAL_PHRASES = ('cannot find this information',)

MIN_OVERLAP_RATIO = 0.4

_NUMBER_RE = re.compile(r'\d+(?:\.\d+)?')
# "1." or "12." followed by a space, and not the end of a decimal, might be
# list numbering. It only counts as numbering when the next or previous
# number in the sequence is also there, so "RR 26. No chest pain" keeps 26
# as a value while "daily 2. Cephalexin" after a "1." is numbering.
_MARKER_CANDIDATE_RE = re.compile(r'(?<![\d.,])\b(\d{1,2})\.(?=\s|$)')

# Words too common near numbers to tie a value to its subject.
_GENERIC_ANCHORS = {
    'daily', 'day', 'days', 'week', 'weeks', 'hour', 'hours', 'every', 'times',
    'twice', 'once', 'dose', 'doses', 'unit', 'units', 'tablet', 'tablets',
    'patient', 'had', 'has', 'have', 'take', 'takes', 'taking', 'given', 'more',
    'total', 'about', 'than', 'then', 'each', 'per', 'level', 'value', 'count',
}
# Chart abbreviations: an answer says "respiratory rate 26" where the note
# says "RR 26". An anchor also matches its abbreviations in the passages.
_ABBREVIATIONS = {
    'respiratory': {'rr'}, 'pressure': {'bp', 'sbp', 'dbp'}, 'blood': {'bp'},
    'heart': {'hr'}, 'pulse': {'hr', 'p'}, 'oxygen': {'o', 'sat', 'spo', 'sao'},
    'saturation': {'sat', 'spo', 'sao'}, 'temperature': {'temp', 't'},
    'white': {'wbc'}, 'hemoglobin': {'hgb', 'hb'}, 'hematocrit': {'hct'},
    'platelets': {'plt'}, 'platelet': {'plt'}, 'potassium': {'k'}, 'sodium': {'na'},
    'chloride': {'cl'}, 'bicarbonate': {'hco', 'bicarb'}, 'creatinine': {'cr', 'creat'},
    'glucose': {'glu', 'bg'}, 'troponin': {'trop', 'tnt'}, 'magnesium': {'mg'},
    'calcium': {'ca'}, 'lactate': {'lac'}, 'nitrogen': {'bun'}, 'urea': {'bun'},
}

# Pre-negation cues and scope terminators, a small NegEx style list.
_NEGATION_CUES = re.compile(
    r'\b(?:no|not|denies|denied|deny|denying|without|never|none|negative for|absence of|free of)\b',
    re.IGNORECASE)
_SCOPE_END = re.compile(r'[,;:.!?\n]|\b(?:but|however|although|except|aside from)\b', re.IGNORECASE)
NEGATION_SCOPE_WORDS = 6
# Words that describe how a finding was recorded rather than the finding
# itself: "no melena reported" does not make "reported" an absent finding.
_NOT_FINDINGS = {
    'reported', 'reports', 'noted', 'present', 'seen', 'found', 'documented', 'observed',
    'identified', 'shown', 'recorded', 'detected', 'further', 'episodes', 'episode',
    'evidence', 'signs', 'history', 'acute', 'new', 'significant', 'other', 'any',
}


def _chunk_text(chunk):
    return chunk['chunk_text'] if isinstance(chunk, dict) else chunk


def _content_words(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _numbers(text):
    return {m.group() for m in value_matches(text)}


def _list_markers(text, known=frozenset()):
    """Matches of list numbering in text: candidates whose neighbour in the
    sequence (k - 1 or k + 1) is also a candidate, or, for an answer that
    quotes one item of a list, a candidate before a capitalised word whose
    number the passages use as list numbering (known)."""
    candidates = list(_MARKER_CANDIDATE_RE.finditer(text))
    present = {int(m.group(1)) for m in candidates}

    def is_marker(m):
        k = int(m.group(1))
        if k - 1 in present or k + 1 in present:
            return True
        return k in known and re.match(r'\s+[A-Z(\[]', text[m.end():]) is not None

    return [m for m in candidates if is_marker(m)]


def _without_markers(text, replacement=' ', known=frozenset()):
    for m in reversed(_list_markers(text, known)):
        text = text[:m.start()] + replacement * (1 if replacement == '\n' else len(m.group())) + text[m.end():]
    return text


def value_matches(text):
    """Number matches in text that are values, with list numbering skipped."""
    markers = {m.start() for m in _list_markers(text)}
    return [m for m in _NUMBER_RE.finditer(text) if m.start() not in markers]


def _split_claims(answer):
    parts = re.split(r'(?<=[.!?])\s+', answer.strip())
    return [p.strip() for p in parts if p.strip()]


def _tokens(text):
    return re.findall(r'[a-z]+|\d+(?:\.\d+)?', text.lower())


def _anchor_word(token):
    return token.isalpha() and len(token) > 2 and token not in _STOPWORDS and token not in _GENERIC_ANCHORS


def _clauses(claim):
    """A claim split at commas, semicolons and line breaks, so each value
    keeps the words of its own clause: "sodium 136, creatinine 0.9" ties
    136 to sodium only, and each line of a quoted list stays separate."""
    return [part for part in re.split(r'[,;\n]', claim) if part.strip()]


def _number_is_placed(number, anchors, context_clauses, context_token_set):
    """False only when the claim ties the number to words the passages use,
    and no clause of the passages holds the number with any of those words."""
    expanded = set(anchors)
    for anchor in anchors:
        expanded |= _ABBREVIATIONS.get(anchor, set())
    known = expanded & context_token_set
    if not known:
        return True  # the claim names the value in its own words; nothing to compare
    return any(number in clause and known & clause for clause in context_clauses)


def _polarity(text):
    """(asserted, negated) content words, with each negation cue scoping the
    next few words up to punctuation or a contrast word."""
    negated_spans = []
    for cue in _NEGATION_CUES.finditer(text):
        rest = text[cue.end():]
        end = _SCOPE_END.search(rest)
        scope = rest[:end.start()] if end else rest
        words = re.findall(r'[a-zA-Z0-9]+', scope)[:NEGATION_SCOPE_WORDS]
        negated_spans.append((cue.start(), cue.end() + len(scope), words))
    negated = {w.lower() for _, _, words in negated_spans for w in words}
    outside = text
    for start, end, _ in sorted(negated_spans, reverse=True):
        outside = outside[:start] + ' ' + outside[end:]
    return _content_words(outside), _content_words(' '.join(negated))


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
    context_token_set = set(_tokens(context_text))
    # Passage clauses for tying values to their subjects. List numbering both
    # splits clauses (chunks often hold a whole list on one line) and is
    # dropped, so "1. Amlodipine 5 mg" does not put 1 next to amlodipine.
    context_clauses = [
        set(_tokens(part))
        for part in re.split(r'[,;\n]|\.\s+(?=[A-Z\[])', _without_markers(context_text, '\n'))
    ]
    context_asserted, context_negated = _polarity(context_text)
    only_absent = context_negated - context_asserted - _NOT_FINDINGS

    # List numbering is removed before the answer is split into sentences,
    # where a marker cut from its item would read as a bare value.
    context_markers = frozenset(int(m.group(1)) for m in _list_markers(context_text))
    draft = _without_markers(draft_answer, known=context_markers)

    unsupported = []
    for claim in _split_claims(draft):
        claim_numbers = _numbers(claim)
        if not claim_numbers <= context_numbers:
            unsupported.append(claim)
            continue

        claim_words = _content_words(claim)
        if not claim_words:
            continue
        if len(claim_words & context_words) / len(claim_words) < MIN_OVERLAP_RATIO:
            unsupported.append(claim)
            continue

        misplaced = False
        for clause in _clauses(claim):
            clause_numbers = _numbers(clause)
            anchors = {t for t in _tokens(clause) if _anchor_word(t)}
            if any(not _number_is_placed(n, anchors, context_clauses, context_token_set) for n in clause_numbers):
                misplaced = True
                break
        claim_asserted, _ = _polarity(claim)
        if misplaced or claim_asserted & only_absent:
            unsupported.append(claim)

    return {'supported': len(unsupported) == 0, 'unsupported_claims': unsupported}
