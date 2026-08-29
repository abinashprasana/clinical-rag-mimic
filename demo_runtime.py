"""Lightweight, synthetic-only runtime for the public Vercel demo.

The full local application uses SentenceTransformers, FAISS, FLAN-T5, and
LangGraph. Those dependencies are intentionally excluded from the public
serverless deployment: their model downloads exceed Vercel's writable
filesystem allowance and the process-local graph memory is not reliable
across function instances.

This module searches only the fully fabricated notes in
``scripts.synthetic_demo_notes`` and returns extractive answers. It performs
no network requests and imports no model, vector, or restricted-data code.
"""

from __future__ import annotations

from functools import lru_cache
import re
from typing import Iterable

from scripts.synthetic_demo_notes import DEMO_NOTES


PUBLIC_DATASET_LABEL = (
    "Synthetic demo (fabricated notes; no patient data; real MIMIC-IV-Note "
    "data requires credentialed PhysioNet access)"
)
PUBLIC_GENERATOR_LABEL = "Deterministic extractive demo"

# Measured by scripts/evaluate_public_demo.py against the same 10-question
# keyword-hit set evaluate_demo.py uses, run directly against this module's
# run_turn(). Not computed live at request time: importing evaluate_demo.py
# here would pull in sentence-transformers/FAISS/FLAN-T5, exactly the heavy
# dependencies this module exists to avoid on a Vercel cold start. Re-run
# that script and update this constant if _score_chunk, _select_evidence, or
# DEMO_NOTES change.
PUBLIC_ACCURACY_PCT = 100

_STOPWORDS = {
    "a", "about", "after", "all", "an", "and", "are", "as", "at", "be",
    "by", "did", "do", "does", "for", "from", "how", "i", "in", "is",
    "it", "me", "of", "on", "or", "please", "record", "show", "that",
    "the", "their", "there", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "with",
}

_INTENT_SECTIONS = (
    (("diagnos", "problem"), ("Discharge Diagnosis",)),
    (("medication", "medicine", "prescrib", "drug"),
     ("Discharge Medications", "Medications on Admission")),
    (("follow", "appointment", "recheck"), ("Followup Instructions",)),
    (("disposition", "discharged to", "facility"), ("Discharge Disposition",)),
    (("condition", "stable", "status"), ("Discharge Condition",)),
    (("allerg",), ("Allergies",)),
    (("lab", "result", "wbc", "creatinine", "sodium", "hemoglobin", "glucose"),
     ("Pertinent Results",)),
    (("vital", "exam", "oxygen", "temperature", "blood pressure"),
     ("Physical Exam",)),
    (("history", "medical history"), ("Past Medical History",)),
    (("complaint", "presented", "presentation"),
     ("Chief Complaint", "History of Present Illness")),
    (("course", "hospital stay", "treated"), ("Brief Hospital Course",)),
    (("instruction", "return", "warning"), ("Discharge Instructions",)),
)

_DOSING_ADVICE_TERMS = (
    "dose", "dosage", "how much", "how often should", "safe amount",
    "recommend a medication", "what should i take",
)


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _matching_sections(question: str) -> set[str]:
    lowered = question.lower()
    matches: set[str] = set()
    for needles, sections in _INTENT_SECTIONS:
        if any(needle in lowered for needle in needles):
            matches.update(sections)
    return matches


def _split_sections(note_text: str) -> Iterable[tuple[str, str]]:
    heading = None
    body: list[str] = []

    for raw_line in note_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(":") and len(line) <= 80:
            if heading is not None and body:
                yield heading, " ".join(body)
            heading = line[:-1]
            body = []
        elif heading is not None:
            body.append(line)

    if heading is not None and body:
        yield heading, " ".join(body)


@lru_cache(maxsize=1)
def _corpus() -> tuple[dict, ...]:
    chunks = []
    chunk_index = 0
    for note in DEMO_NOTES:
        for heading, body in _split_sections(note["text"]):
            chunk_text = f"{heading}: {body}"
            chunks.append({
                "subject_id": note["subject_id"],
                "hadm_id": note["hadm_id"],
                "chunk_idx": chunk_index,
                "section": heading,
                "chunk_text": chunk_text,
                "tokens": _tokens(chunk_text),
            })
            chunk_index += 1
    return tuple(chunks)


def _score_chunk(question: str, question_tokens: set[str], requested: set[str], chunk: dict) -> float:
    overlap = len(question_tokens & chunk["tokens"])
    score = float(overlap * 2)
    if requested and chunk["section"] in requested:
        score += 12.0

    lowered = question.lower()
    if str(chunk["subject_id"]) in lowered:
        score += 30.0
    if str(chunk["hadm_id"]) in lowered:
        score += 30.0

    section_tokens = _tokens(chunk["section"])
    score += len(question_tokens & section_tokens) * 3.0
    return score


def _select_evidence(question: str, limit: int = 5) -> list[dict]:
    question_tokens = _tokens(question)
    requested = _matching_sections(question)
    ranked = []

    for chunk in _corpus():
        score = _score_chunk(question, question_tokens, requested, chunk)
        if score > 0:
            ranked.append((score, chunk))

    ranked.sort(
        key=lambda item: (
            -item[0], item[1]["subject_id"], item[1]["chunk_idx"]
        )
    )

    selected = []
    seen_notes = set()
    for score, chunk in ranked:
        note_key = (chunk["subject_id"], chunk["hadm_id"])
        if note_key in seen_notes and len(selected) < min(3, limit):
            continue
        seen_notes.add(note_key)
        selected.append({
            "subject_id": chunk["subject_id"],
            "hadm_id": chunk["hadm_id"],
            "chunk_idx": chunk["chunk_idx"],
            "score": round(score / (score + 12.0), 4),
            "chunk_text": chunk["chunk_text"],
            "section": chunk["section"],
        })
        if len(selected) == limit:
            break
    return selected


def _answer_from_evidence(evidence: list[dict]) -> str:
    primary = evidence[0]
    body = primary["chunk_text"].split(":", 1)[1].strip()
    return (
        "In the highest-matching fabricated demo record "
        f"(subject {primary['subject_id']}, admission {primary['hadm_id']}), "
        f"the documented {primary['section'].lower()} is: {body}"
    )


def run_turn(question: str, thread_id: str | None = None) -> dict:
    """Return the same response contract as the full agent graph.

    ``thread_id`` is accepted for API compatibility, but this public runtime
    is intentionally stateless so it behaves consistently across serverless
    instances.
    """
    del thread_id
    normalized = question.strip()
    lowered = normalized.lower()

    if any(term in lowered for term in _DOSING_ADVICE_TERMS):
        return {
            "final_answer": (
                "This public demo does not provide prescribing or dosing advice. "
                "It can quote medications documented in its fabricated records; "
                "use an official drug label and a qualified clinician for dosing decisions."
            ),
            "route": "direct",
            "tool_used": "Public-demo safety boundary",
            "citations": [],
            "reflection": {"supported": True, "unsupported_claims": []},
            "needs_clarification": False,
            "fda_result": None,
        }

    evidence = _select_evidence(normalized)
    if not evidence:
        return {
            "final_answer": (
                "I could not match that question to the fabricated demo records. "
                "Try asking about a discharge diagnosis, medication, condition, "
                "disposition, allergy, lab result, or follow-up instruction."
            ),
            "route": "clarify",
            "tool_used": "Synthetic demo clarification",
            "citations": [],
            "reflection": None,
            "needs_clarification": True,
            "fda_result": None,
        }

    return {
        "final_answer": _answer_from_evidence(evidence),
        "route": "retrieve",
        "tool_used": f"Retrieved {len(evidence)} fabricated evidence passages",
        "citations": [
            {key: item[key] for key in (
                "subject_id", "hadm_id", "chunk_idx", "score", "chunk_text"
            )}
            for item in evidence
        ],
        "reflection": {
            "supported": True,
            "unsupported_claims": [],
            "method": "deterministic extractive response",
        },
        "needs_clarification": False,
        "fda_result": None,
    }
