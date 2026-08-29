"""Shared state schema for the LangGraph agent. One AgentState flows through
every node in a turn; LangGraph's MemorySaver checkpoints it per session
(thread_id), so `messages`/prior turns persist across a browser session but
never to disk (see agent/graph.py)."""
from typing import TypedDict, Optional


class RetrievedChunk(TypedDict):
    chunk_text: str
    subject_id: Optional[int]
    hadm_id: Optional[int]
    chunk_idx: int
    score: float


class Reflection(TypedDict, total=False):
    supported: bool
    unsupported_claims: list[str]
    structural_note: Optional[str]   # from the optional external structural check
    well_formed: Optional[bool]


class AgentState(TypedDict, total=False):
    session_id: str
    messages: list[dict]                       # {role, content, ts}
    question: str
    route: Optional[str]                       # "retrieve" | "dosage" | "clarify" | "direct"
    drug_name: Optional[str]
    retrieved_chunks: list[RetrievedChunk]
    fda_result: Optional[dict]
    draft_answer: Optional[str]
    reflection: Optional[Reflection]
    reflection_regenerated: bool                # true once one retry has been used
    final_answer: Optional[str]
    citations: list[dict]
    tool_used: Optional[str]                    # what respond_node reports to the UI
    needs_clarification: bool
    clarification_question: Optional[str]
    step_count: int
    error: Optional[str]
