"""Synthetic corruption stress set for the faithfulness gate.

Supported answers are corrupted by rule based transformations in the style
of FactCC (Kryscinski et al. 2020). Every corrupted answer is unsupported by
construction, so the gate should flag it. Results are reported as their own
rows, marked synthetic, and never blended with the human label numbers.

* number_swap_absent      a number is replaced by one that appears nowhere in the context
* number_swap_in_context  a number is replaced by a different number that does appear in the context
* insert_fact             an invented clinical sentence is appended
* drop_negation           negation words (no, not, denies, without, ...) are removed
"""
import re

from agent.reflection import local_reflect, value_matches

CORRUPTIONS = ('number_swap_absent', 'number_swap_in_context', 'insert_fact', 'drop_negation')

_NUMBER_RE = re.compile(r'\d+(?:\.\d+)?')
_NEGATION_RE = re.compile(
    r'\b(?:no|not|denies|denied|without|never|negative for)\b\s*', re.IGNORECASE)
# Invented statements used only to corrupt answers. The first one whose
# numbers and distinctive words are absent from the context is used.
INVENTED_FACTS = (
    'The patient also underwent coronary artery bypass grafting with three grafts.',
    'A lumbar puncture showed an opening pressure of 34 cm of water.',
    'The patient was started on warfarin 7.5 mg nightly for a mechanical valve.',
    'Magnetic resonance imaging revealed a 2.3 cm pituitary adenoma.',
)


def _numbers(text):
    return _NUMBER_RE.findall(text)


# Anything that might be list numbering ("1." at the start or before a
# capitalised word) is left alone, so a swap always changes a real value.
_MAYBE_MARKER_RE = re.compile(r'(?:^|(?<=\n)|(?<![\d.,])\b)\d{1,2}\.(?=\s+[A-Z(\[]|$)', re.MULTILINE)


def _value_matches(text):
    maybe = {m.start() for m in _MAYBE_MARKER_RE.finditer(text)}
    return [m for m in value_matches(text) if m.start() not in maybe]


def _swap_first_number(draft, replacement_for):
    """Replaces the first clinical number (list markers skipped) with
    replacement_for(value). A replacement that equals the value numerically,
    such as 5 for 5.0, is not a corruption and is skipped."""
    for match in _value_matches(draft):
        new = replacement_for(match.group())
        if new is None or float(new) == float(match.group()):
            continue
        return draft[:match.start()] + new + draft[match.end():]
    return None


def corrupt(draft, context_text):
    """{corruption type: corrupted text} for the corruptions that apply."""
    context_numbers = set(_numbers(context_text))
    out = {}

    def absent(value):
        candidate = int(float(value)) * 3 + 7
        while str(candidate) in context_numbers:
            candidate += 11
        return str(candidate)

    # Another clinical value from the same passages (list markers excluded),
    # numerically different from the one it replaces.
    context_values = sorted({m.group() for m in _value_matches(context_text)})

    def present(value):
        others = [n for n in context_values if float(n) != float(value)]
        return others[0] if others else None

    for name, fn in (('number_swap_absent', absent), ('number_swap_in_context', present)):
        corrupted = _swap_first_number(draft, fn)
        if corrupted:
            out[name] = corrupted

    lowered = context_text.lower()
    for fact in INVENTED_FACTS:
        if not any(n in context_numbers for n in _numbers(fact)) and fact.lower() not in lowered:
            out['insert_fact'] = draft.rstrip() + ' ' + fact
            break

    if _NEGATION_RE.search(draft):
        out['drop_negation'] = re.sub(r'\s+', ' ', _NEGATION_RE.sub('', draft)).strip()
    return out


def run_stress(items):
    """items: iterable of (draft, passages as a list of chunk texts).
    Returns {type: {'n': applicable, 'detected': flagged by the gate}}."""
    counts = {name: {'n': 0, 'detected': 0} for name in CORRUPTIONS}
    for draft, passages in items:
        if local_reflect(draft, passages)['supported'] is not True:
            continue  # the source answer must pass the gate, or detection is meaningless
        for name, corrupted in corrupt(draft, ' '.join(passages)).items():
            counts[name]['n'] += 1
            counts[name]['detected'] += not local_reflect(corrupted, passages)['supported']
    return counts
