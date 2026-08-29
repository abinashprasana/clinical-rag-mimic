"""Retrieval-augmented runtime for the public Vercel demo, backed by Gemini's
free-tier API instead of locally-loaded models.

The full local application uses SentenceTransformers, FAISS, FLAN-T5, and
LangGraph running in-process. Those specific packages are excluded from the
public serverless deployment because their combined size and model weights
exceed Vercel's function size limit -- not because retrieval-augmented
generation itself can't run there. Vercel functions can make outbound HTTP
calls just fine, so this module gets the same two model calls (embed the
question, generate the answer) from Gemini's hosted API instead of loading
the models in-process, keeping the deployed function small while still doing
real retrieval against the same 101 synthetic chunks
(outputs_demo/chunks_data.pkl, outputs_demo/gemini_chunk_embeddings.pkl) the
local pipeline uses, and running the answer through the same local
faithfulness check (agent.reflection.local_reflect) the local pipeline uses.

If GEMINI_API_KEY isn't configured, or any Gemini call fails for any reason
(quota, network, timeout), this module falls back to a fully offline,
deterministic keyword-matching method that never leaves the process -- the
app stays functional either way, just cruder without a key, mirroring how
agent/llm.py already handles a missing key for routing/reflection.
"""

from __future__ import annotations

from functools import lru_cache
import pickle
import re
from typing import Iterable

import numpy as np
from google import genai
from google.genai import types

import config
from agent.reflection import local_reflect
from scripts.synthetic_demo_notes import DEMO_NOTES

EMBEDDING_MODEL = 'models/gemini-embedding-001'
EMBEDDINGS_PATH = 'outputs_demo/gemini_chunk_embeddings.pkl'
_TIMEOUT_MS = config.GEMINI_TIMEOUT_SECONDS * 1000

# Deliberately NOT config.GEMINI_MODEL: that setting (gemini-3.6-flash by
# default) is tuned for the local app's own low-volume routing/reflection
# calls, whose free tier caps generateContent at just 20 requests/day --
# confirmed by hitting that exact limit during development. A public demo
# answering anonymous visitors' questions needs a model with a free daily
# quota built for sustained volume; gemini-flash-lite-latest's free tier
# (~1,000-1,500 requests/day as of 2026) is the right fit for that, and is
# billed on a completely separate quota from GEMINI_MODEL.
PUBLIC_GENERATION_MODEL = 'models/gemini-flash-lite-latest'

GENERATION_SYSTEM_PROMPT = """You are a clinical assistant answering questions about hospital discharge notes \
covering a range of clinical conditions.
Answer the question below using only the context provided.
Include every distinct item the context mentions that is relevant to the question.
Use exact medical terms, doses, and values from the context, but write the answer \
in your own words as complete sentences -- never copy numbering or list markers \
from the context, and never state the same fact twice.
Be specific and complete rather than brief.
If the answer is not in the context, say: I cannot find this information in the provided notes."""

PUBLIC_DATASET_LABEL = (
    "Synthetic demo (fabricated notes; no patient data; real MIMIC-IV-Note "
    "data requires credentialed PhysioNet access)"
)
PUBLIC_GENERATOR_LABEL = f"{PUBLIC_GENERATION_MODEL.removeprefix('models/')} (retrieval-augmented, local fallback)"

# Measured by scripts/evaluate_public_demo.py against the same 10-question
# keyword-hit set evaluate_demo.py uses, run directly against this module's
# run_turn() with GEMINI_API_KEY set, so this reflects the real
# retrieval-augmented (Gemini embeddings + generation, header-boosted
# retrieval, local faithfulness check) path, not just the offline fallback.
# Re-run after changing GENERATION_SYSTEM_PROMPT, EMBEDDING_MODEL, the
# header-boost weight, or DEMO_NOTES.
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
    """Verbatim fallback shared by both the offline keyword path and the
    Gemini path's failed-generation fallback. The keyword path's evidence
    dicts include an explicit 'section' key; the Gemini path's don't, so the
    section name is derived from the chunk's own "Header: body" text either
    way -- this must work for both, not assume the keyword-only shape."""
    primary = evidence[0]
    header, _, body = primary["chunk_text"].partition(":")
    section = primary.get("section", header.strip())
    return (
        "In the highest-matching fabricated demo record "
        f"(subject {primary['subject_id']}, admission {primary['hadm_id']}), "
        f"the documented {section.lower()} is: {body.strip()}"
    )


_client = None
_client_checked = False


def get_client():
    """Mirrors agent.llm.get_client(): returns None (never raises) if no key
    is configured, so callers fall back to the offline path automatically."""
    global _client, _client_checked
    if _client_checked:
        return _client
    _client_checked = True
    if config.GEMINI_API_KEY:
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


@lru_cache(maxsize=1)
def _semantic_corpus():
    """Loads the precomputed Gemini embeddings for the synthetic chunks (see
    scripts/build_demo_gemini_embeddings.py). Raises if the file is missing
    -- callers must catch that alongside network errors, since a missing
    file is exactly as much reason to fall back as a failed API call."""
    with open(EMBEDDINGS_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['chunks'], data['provenance'], data['embeddings']


_HEADER_RE = re.compile(r"^\[([^\]]+)\]")


def _header_boost(chunk_text: str, question_tokens: set[str], weight: float = 0.15) -> float:
    """Same technique retrieval.py's _header_boost uses for the real FAISS
    pipeline: pure cosine similarity on this small, repetitive-structure
    corpus can rank a near-topic section (e.g. "Discharge Condition") above
    the actually-asked-about one (e.g. "Discharge Diagnosis") since both
    embed similarly as short clinical-outcome phrases. Rewarding a chunk
    whose own header lexically matches the question corrects that without
    another model call."""
    match = _HEADER_RE.match(chunk_text)
    if not match:
        return 0.0
    header_tokens = _tokens(match.group(1))
    if not header_tokens:
        return 0.0
    return weight * (len(header_tokens & question_tokens) / len(header_tokens))


def _semantic_retrieve(client, question: str, k: int = 5) -> list[dict] | None:
    """Real retrieval: embeds the question via Gemini, cosine-matches
    against the precomputed chunk embeddings, re-ranked by the same
    header-relevance boost the local pipeline uses. Returns None (not an
    empty list) on any failure so callers can distinguish "found nothing"
    from "couldn't even try"."""
    try:
        chunks, provenance, embeddings = _semantic_corpus()
        resp = client.models.embed_content(model=EMBEDDING_MODEL, contents=[question])
        query_vec = np.array(resp.embeddings[0].values, dtype='float32')
        query_vec /= max(np.linalg.norm(query_vec), 1e-9)
        cosine_scores = embeddings @ query_vec
        question_tokens = _tokens(question)
        boosted = np.array([
            cosine_scores[i] + _header_boost(chunks[i], question_tokens)
            for i in range(len(chunks))
        ])
        top_idx = np.argsort(-boosted)[:k]
        return [
            {
                'subject_id': provenance[i]['subject_id'],
                'hadm_id': provenance[i]['hadm_id'],
                'chunk_idx': int(i),
                'score': round(float(cosine_scores[i]), 4),
                'chunk_text': chunks[i],
            }
            for i in top_idx
        ]
    except Exception:
        return None


def _generate_with_gemini(client, question: str, evidence: list[dict], strict: bool = False) -> str | None:
    context = '\n\n'.join(f'[Passage {i + 1}]\n{item["chunk_text"]}' for i, item in enumerate(evidence))
    prompt = f'Context:\n{context}\n\nQuestion: {question}'
    if strict:
        prompt += (
            '\n\nYour previous answer included claims not directly supported by the '
            'context above. Answer again using ONLY facts stated verbatim in the context.'
        )
    try:
        resp = client.models.generate_content(
            model=PUBLIC_GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=GENERATION_SYSTEM_PROMPT,
                http_options=types.HttpOptions(timeout=_TIMEOUT_MS),
            ),
        )
        return resp.text.strip() if resp.text else None
    except Exception:
        return None


def _gemini_turn(client, question: str) -> dict | None:
    """Attempts the full retrieve-then-generate path via Gemini. Returns None
    on any failure at any step so run_turn can fall back to the fully
    offline path without a partial/inconsistent result."""
    evidence = _semantic_retrieve(client, question)
    if not evidence:
        return None

    draft = _generate_with_gemini(client, question, evidence)
    if not draft:
        return None

    reflection = local_reflect(draft, evidence)
    if not reflection['supported']:
        retry_draft = _generate_with_gemini(client, question, evidence, strict=True)
        if retry_draft:
            retry_reflection = local_reflect(retry_draft, evidence)
            if retry_reflection['supported']:
                draft, reflection = retry_draft, retry_reflection

    if not reflection['supported']:
        # Never publish a generated claim that failed its own faithfulness
        # check -- fall back to a verbatim quote from the same real
        # semantic-retrieval evidence instead of a possibly-fabricated draft.
        return {
            'final_answer': _answer_from_evidence(evidence),
            'route': 'retrieve',
            'tool_used': f'Retrieved {len(evidence)} passages via Gemini embeddings (generation unsupported, quoted directly)',
            'citations': evidence,
            'reflection': {'supported': True, 'unsupported_claims': [], 'method': 'verbatim fallback after failed generation'},
            'needs_clarification': False,
            'fda_result': None,
        }

    return {
        'final_answer': draft,
        'route': 'retrieve',
        'tool_used': f'Retrieved {len(evidence)} passages via Gemini embeddings, generated by {PUBLIC_GENERATION_MODEL.removeprefix("models/")}',
        'citations': evidence,
        'reflection': {**reflection, 'method': 'local content-word overlap check on Gemini draft'},
        'needs_clarification': False,
        'fda_result': None,
    }


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

    client = get_client()
    if client is not None:
        result = _gemini_turn(client, normalized)
        if result is not None:
            return result

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
        "tool_used": f"Retrieved {len(evidence)} fabricated evidence passages (offline keyword fallback)",
        "citations": [
            {key: item[key] for key in (
                "subject_id", "hadm_id", "chunk_idx", "score", "chunk_text"
            )}
            for item in evidence
        ],
        "reflection": {
            "supported": True,
            "unsupported_claims": [],
            "method": "deterministic extractive response (offline fallback, no Gemini call)",
        },
        "needs_clarification": False,
        "fda_result": None,
    }
