"""LangGraph orchestration: retrieve -> decide-if-tool-needed -> call tool ->
reflect -> respond. Nodes marked STUB below are placeholder logic validating
the graph wiring/checkpointing/session-keying; they're replaced with real
Gemini/local-model calls in later build steps without changing the graph
shape itself.

Session memory: MemorySaver checkpoints AgentState per thread_id, in-process
only, no disk persistence -- state is gone on restart, by design ("session-
level memory only"). A single module-level graph/checkpointer instance is
reused across requests, so Flask must run as a single worker process (see
app.py's waitress config) or threads would each get their own state.
"""
import re

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.tools import fda_label_tool
from agent.reflection import local_reflect
from agent import llm
from retrieval import retrieve_chunks
from generation import generate_answer
import config

REFUSAL_TEXT = (
    "I could not confirm this answer is fully supported by the retrieved "
    "notes -- please review the cited excerpts directly."
)

_STOPWORDS = {
    'the', 'a', 'an', 'is', 'of', 'for', 'and', 'what', 'dose', 'dosage',
    'contraindications', 'contraindication', 'to', 'in', 'take', 'much',
    'how', 'mg', 'daily', 'does', 'do', 'i', 'my',
}


def _guess_drug_name(question: str) -> str | None:
    """Crude candidate-word extraction, replaced by Gemini structured
    extraction in the Gemini-integration build step. Picks the first
    non-stopword token as a placeholder guess."""
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", question)
    for w in words:
        if w.lower() not in _STOPWORDS:
            return w
    return None


# --- Nodes -------------------------------------------------------------

def _stub_route(state: AgentState) -> None:
    """Fallback used when no GEMINI_API_KEY is configured. Cruder than the
    Gemini router but keeps the app functional without a key."""
    question = state['question'].lower()
    dosage_words = ('dose', 'dosage', 'contraindicat', 'mg ', 'milligram', 'overdose')
    if any(w in question for w in dosage_words):
        state['route'] = 'dosage'
        state['drug_name'] = _guess_drug_name(state['question'])
    elif len(question.split()) < 3:
        state['route'] = 'clarify'
        state['clarification_question'] = (
            "Could you say a bit more about what you'd like to know? "
            "For example, ask about a specific note's diagnosis, medications, "
            "or a drug's dosage/contraindications."
        )
    else:
        state['route'] = 'retrieve'


def route_node(state: AgentState) -> AgentState:
    """Routes via Gemini structured output when a free-tier API key is
    configured; otherwise falls back to a local keyword heuristic. Only ever
    sees the question + short history, never clinical note text."""
    state['step_count'] = state.get('step_count', 0) + 1
    decision = llm.gemini_route(state['question'], state.get('messages'))
    if decision is None:
        _stub_route(state)
        return state

    state['route'] = decision.get('route', 'retrieve')
    if state['route'] == 'dosage':
        state['drug_name'] = decision.get('drug_name') or _guess_drug_name(state['question'])
    if state['route'] == 'clarify':
        state['clarification_question'] = (
            decision.get('clarification_question')
            or "Could you say a bit more about what you'd like to know?"
        )
    return state


def make_retrieve_node(embedding_model, faiss_index, chunks, provenance):
    """Real MIMIC retrieval tool -- not an LLM call, so it's wired for real
    now rather than stubbed. Injected with the heavy FAISS/model objects
    loaded once at app startup rather than reloading them per call."""
    def retrieve_node(state: AgentState) -> AgentState:
        state['step_count'] += 1
        state['retrieved_chunks'] = retrieve_chunks(
            state['question'], embedding_model, faiss_index, chunks, provenance,
            k=config.DEFAULT_TOP_K,
        )
        return state
    return retrieve_node


def dosage_node(state: AgentState) -> AgentState:
    """Real openFDA tool call -- not an LLM call, so it's wired for real now.
    Returns None (never a fabricated answer) when the drug isn't found."""
    state['step_count'] += 1
    drug_name = state.get('drug_name')
    state['fda_result'] = fda_label_tool(drug_name) if drug_name else None
    return state


def clarify_node(state: AgentState) -> AgentState:
    state['step_count'] += 1
    state['needs_clarification'] = True
    if not state.get('clarification_question'):
        phrased = llm.gemini_clarify(state['question'])
        state['clarification_question'] = phrased or "Could you clarify your question?"
    return state


def make_generate_node(local_generator):
    """Real local-model generation. Dosage answers never go through the
    local model by default -- FDA label text renders directly as a
    reference card, avoiding paraphrase-introduced error on safety-critical
    dosage text."""
    def generate_node(state: AgentState) -> AgentState:
        state['step_count'] += 1
        if state.get('fda_result'):
            state['draft_answer'] = None
            state['tool_used'] = f"FDA Label Lookup: {state['fda_result'].get('drug_name')}"
        elif state.get('retrieved_chunks'):
            answer, _ = generate_answer(state['question'], state['retrieved_chunks'], local_generator)
            state['draft_answer'] = answer
            state['tool_used'] = f"Retrieved {len(state['retrieved_chunks'])} evidence passages"
        else:
            answer, _ = generate_answer(state['question'], [], local_generator)
            state['draft_answer'] = answer
            state['tool_used'] = 'Answered directly'
        return state
    return generate_node


def reflect_node(state: AgentState) -> AgentState:
    """Local faithfulness check -- does the draft answer only make claims
    supported by the retrieved chunks? Fully local, never calls Gemini; this
    is the real safety gate (see agent/reflection.py)."""
    state['step_count'] += 1
    state['reflection'] = local_reflect(state.get('draft_answer'), state.get('retrieved_chunks', []))
    return state


def reflect_structure_node(state: AgentState) -> AgentState:
    """Advisory structural check only -- the local reflect_node already
    passed the faithfulness check by the time this runs, so this never
    forces a refusal on its own, it only annotates. Sends Gemini the
    question and the model's own draft answer, never raw retrieved chunks
    -- see the plan's reflection privacy note."""
    state['step_count'] += 1
    result = llm.gemini_reflect_structure(state['question'], state.get('draft_answer') or '')
    if result is not None and state.get('reflection'):
        state['reflection']['structural_note'] = result.get('note')
        state['reflection']['well_formed'] = result.get('well_formed', True)
    return state


def respond_node(state: AgentState) -> AgentState:
    reflection = state.get('reflection') or {'supported': True, 'unsupported_claims': []}

    if state.get('needs_clarification'):
        final_answer = state['clarification_question']
        tool_used = 'Clarification requested'
    elif state.get('fda_result'):
        final_answer = None  # UI renders the FDA card directly, not as chat prose
        tool_used = state.get('tool_used')
    elif not reflection['supported']:
        final_answer = REFUSAL_TEXT
        tool_used = state.get('tool_used')
    else:
        final_answer = state.get('draft_answer')
        tool_used = state.get('tool_used')

    state['final_answer'] = final_answer
    state['tool_used'] = tool_used
    state['citations'] = [
        {
            'subject_id': c['subject_id'],
            'hadm_id': c['hadm_id'],
            'chunk_idx': c['chunk_idx'],
            'score': round(c['score'], 4),
            'chunk_text': c['chunk_text'],
        }
        for c in state.get('retrieved_chunks', [])
    ]
    state['messages'] = state.get('messages', []) + [
        {'role': 'user', 'content': state['question']},
        {'role': 'assistant', 'content': final_answer or '[FDA label reference returned]'},
    ]
    return state


# --- Routing -------------------------------------------------------------

def _route_decision(state: AgentState) -> str:
    return state['route']


def _needs_reflection(state: AgentState) -> str:
    if state.get('fda_result'):
        return 'skip'
    return 'reflect'


def _reflect_decision(state: AgentState) -> str:
    reflection = state.get('reflection') or {'supported': True}
    if reflection['supported']:
        return 'structure' if config.ENABLE_EXTERNAL_REFLECTION else 'respond'
    if not state.get('reflection_regenerated') and state['step_count'] < config.MAX_AGENT_STEPS:
        state['reflection_regenerated'] = True
        return 'retry'
    return 'respond'  # already retried once, or hit the step cap -- forced refusal in respond_node


def build_graph(embedding_model, faiss_index, chunks, provenance, local_generator):
    graph = StateGraph(AgentState)
    graph.add_node('route', route_node)
    graph.add_node('retrieve', make_retrieve_node(embedding_model, faiss_index, chunks, provenance))
    graph.add_node('dosage', dosage_node)
    graph.add_node('clarify', clarify_node)
    graph.add_node('generate', make_generate_node(local_generator))
    graph.add_node('reflect', reflect_node)
    graph.add_node('reflect_structure', reflect_structure_node)
    graph.add_node('respond', respond_node)

    graph.set_entry_point('route')
    graph.add_conditional_edges('route', _route_decision, {
        'retrieve': 'retrieve',
        'dosage': 'dosage',
        'clarify': 'clarify',
        'direct': 'generate',
    })
    graph.add_edge('retrieve', 'generate')
    graph.add_edge('dosage', 'generate')
    graph.add_conditional_edges('generate', _needs_reflection, {
        'reflect': 'reflect',
        'skip': 'respond',
    })
    graph.add_conditional_edges('reflect', _reflect_decision, {
        'structure': 'reflect_structure',
        'respond': 'respond',
        'retry': 'generate',
    })
    graph.add_edge('reflect_structure', 'respond')
    graph.add_edge('clarify', 'respond')
    graph.add_edge('respond', END)

    return graph.compile(checkpointer=MemorySaver())


# Module-level singleton: one checkpointer/graph instance per process,
# matching the single-worker-process requirement (see docstring above).
_compiled_graph = None


def get_graph(embedding_model=None, faiss_index=None, chunks=None, provenance=None, local_generator=None):
    """Builds the compiled graph on first call and reuses it after. The
    dependencies are only required on the first (building) call -- later
    callers (e.g. Flask request handlers) can call get_graph() with no
    arguments once the app has initialized it once at startup."""
    global _compiled_graph
    if _compiled_graph is None:
        if embedding_model is None:
            raise RuntimeError(
                'get_graph() must be called once with embedding_model/faiss_index/'
                'chunks/provenance/local_generator (e.g. at app startup) before '
                'being called with no arguments.'
            )
        _compiled_graph = build_graph(embedding_model, faiss_index, chunks, provenance, local_generator)
    return _compiled_graph


def run_turn(question: str, thread_id: str) -> AgentState:
    """Invoke one agent turn for a given session's thread_id. MemorySaver
    carries prior turns' state (messages, etc.) forward automatically."""
    graph = get_graph()
    config_arg = {"configurable": {"thread_id": thread_id}}
    initial: AgentState = {
        'question': question,
        'step_count': 0,
        'reflection_regenerated': False,
        'needs_clarification': False,
        'route': None,
        'retrieved_chunks': [],
        'fda_result': None,
    }
    return graph.invoke(initial, config=config_arg)
