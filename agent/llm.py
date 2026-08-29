"""Gemini calls for orchestration only: tool routing, clarification phrasing,
and a structural (not faithfulness) check on the local model's own draft
answer. None of these functions ever receive raw clinical note text -- see
the plan's reflection privacy note. Google Gemini's free tier requires no
credit card (https://aistudio.google.com/apikey).

If no GEMINI_API_KEY is configured, every function here returns None and
callers fall back to a local heuristic -- the app stays fully functional
without a key, just with cruder routing/no structural reflection pass.
"""
import json
from google import genai
from google.genai import types
import config

_client = None
_client_checked = False
_TIMEOUT_MS = config.GEMINI_TIMEOUT_SECONDS * 1000


def get_client():
    global _client, _client_checked
    if _client_checked:
        return _client
    _client_checked = True
    if config.GEMINI_API_KEY:
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["retrieve", "dosage", "clarify", "direct"]},
        "drug_name": {"type": "string", "nullable": True},
        "clarification_question": {"type": "string", "nullable": True},
    },
    "required": ["route"],
}

ROUTE_SYSTEM_PROMPT = """You route questions for a clinical assistant to one of four tools. \
You only ever see the user's question and short conversation history -- never any clinical \
note content. Choose exactly one route:

- "retrieve": the question asks about a specific patient's clinical note/discharge summary \
  (diagnosis, medications, history, labs, vitals, follow-up, etc).
- "dosage": the question asks about a drug's dosage, contraindications, or warnings in \
  general (not about a specific patient's note). Also extract the drug name into drug_name.
- "clarify": the question is too ambiguous or underspecified to route confidently (e.g. \
  missing a drug name for a dosage question, or too vague to know what's being asked). \
  Write a short, specific clarification_question.
- "direct": a meta/chit-chat question about the assistant itself, not requiring any tool.

Respond only with the JSON schema provided."""


def gemini_route(question: str, history: list[dict] | None = None) -> dict | None:
    client = get_client()
    if client is None:
        return None
    history = history or []
    contents = [f"{m['role']}: {m['content']}" for m in history[-6:]]
    contents.append(f"user: {question}")
    try:
        resp = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents="\n".join(contents),
            config=types.GenerateContentConfig(
                system_instruction=ROUTE_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=ROUTE_SCHEMA,
                http_options=types.HttpOptions(timeout=_TIMEOUT_MS),
            ),
        )
        return json.loads(resp.text)
    except Exception:
        return None


def gemini_clarify(question: str, reason: str = "") -> str | None:
    client = get_client()
    if client is None:
        return None
    prompt = (
        "Write one short, specific clarifying question (one sentence) to ask the user so a "
        "clinical assistant can answer their request. Do not answer the question yourself.\n\n"
        f"User's message: {question}\n"
        f"Why it's ambiguous: {reason or 'not specified'}"
    )
    try:
        resp = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                http_options=types.HttpOptions(timeout=_TIMEOUT_MS),
            ),
        )
        return resp.text.strip()
    except Exception:
        return None


STRUCTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "well_formed": {"type": "boolean"},
        "note": {"type": "string"},
    },
    "required": ["well_formed"],
}


def gemini_reflect_structure(question: str, draft_answer: str) -> dict | None:
    """Structural quality check only -- never receives retrieved chunks, only
    the question and the model's own draft answer. This is advisory, not the
    faithfulness gate (that's the local reflect_node check)."""
    client = get_client()
    if client is None:
        return None
    prompt = (
        "You are reviewing an AI-generated draft answer for structural quality only -- "
        "not for factual accuracy (you cannot verify facts, you don't have the source). "
        "Check: does it actually address the question, is it internally consistent, "
        "does it avoid overconfident absolute claims where hedging would be appropriate.\n\n"
        f"Question: {question}\n"
        f"Draft answer: {draft_answer}"
    )
    try:
        resp = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=STRUCTURE_SCHEMA,
                http_options=types.HttpOptions(timeout=_TIMEOUT_MS),
            ),
        )
        return json.loads(resp.text)
    except Exception:
        return None
