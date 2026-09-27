import os
from dotenv import load_dotenv

load_dotenv()


def _bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in ('1', 'true', 'yes', 'on')


# --- Data ---
DATA_PATH = os.path.join(
    'mimic-iv-note-deidentified-free-text-clinical-notes-2.2', 'note', 'discharge.csv.gz'
)
SAMPLE_SIZE = int(os.getenv('SAMPLE_SIZE', 5000))
RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))

# --- Chunking ---
MIMIC_SECTIONS = [
    'Chief Complaint', 'History of Present Illness', 'Past Medical History',
    'Social History', 'Family History', 'Allergies', 'Physical Exam',
    'Pertinent Results', 'Brief Hospital Course', 'Medications on Admission',
    'Discharge Medications', 'Discharge Disposition', 'Discharge Diagnosis',
    'Discharge Condition', 'Discharge Instructions', 'Followup Instructions'
]
MAX_CHUNK_WORDS = int(os.getenv('MAX_CHUNK_WORDS', 180))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 40))

# --- Embeddings / Retrieval ---
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', 64))
DEFAULT_TOP_K = int(os.getenv('DEFAULT_TOP_K', 5))
# Embed each chunk with a short note-level context line in front of it
# (core/chunking.contextual_texts). Changes only the index text, never the
# stored or displayed chunk. Rebuild the index after changing this.
CONTEXTUAL_INDEX = _bool('CONTEXTUAL_INDEX', True)
# Compare Porter stems of question and section header words in the header
# boost, so "discharged" matches "Discharge". Chosen on the evaluation dev
# split; see evals/tune_retrieval.py and the README Evaluation section.
RETRIEVAL_STEM_HEADERS = _bool('RETRIEVAL_STEM_HEADERS', True)
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'outputs/')
# Shown in the UI's "Dataset" field. Set to something like "Synthetic demo
# notes (not real patient data)" when OUTPUT_DIR points at outputs_demo/ --
# see demo/build_index.py -- so a public deployment always honestly
# discloses when it's not running on real MIMIC-IV data.
DATASET_LABEL = os.getenv('DATASET_LABEL', 'MIMIC-IV-Note v2.2')

# --- Generation (local model, always stays on-device) ---
LOCAL_GENERATOR_MODEL = os.getenv('LOCAL_GENERATOR_MODEL', 'google/flan-t5-base')
MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', 200))
# How many of the retrieved passages go into the generator prompt. All
# DEFAULT_TOP_K passages are still cited and used by the faithfulness check.
GENERATION_TOP_K = int(os.getenv('GENERATION_TOP_K', 5))
# Ask the local model whether the passages can answer the question before
# generating, and refuse when it says no (core/generation.is_answerable).
ANSWERABILITY_CHECK = _bool('ANSWERABILITY_CHECK', False)
# How record answers are produced (chosen on the evaluation dev split, see
# evals/tune_hybrid.py):
#   'hybrid'      split a two-part question into parts and retrieve each; a
#                 part whose top ranked passage or best evidence is a numbered
#                 list (a medication list) is answered with that list
#                 verbatim, any other part by the local generator; refuse
#                 only when the answerability
#                 judge says no to every part AND no part's evidence scores
#                 at least EXTRACTIVE_THRESHOLD with the cross encoder.
#   'generative'  the original path: the local generator writes the answer
#                 from the retrieved passages.
ANSWER_MODE = os.getenv('ANSWER_MODE', 'hybrid')
EXTRACTIVE_RERANKER = os.getenv('EXTRACTIVE_RERANKER', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
# Chosen in round 2 on the 100 original questions (results were identical
# from -8 to -6); see evals/tune_round2.py.
EXTRACTIVE_THRESHOLD = float(os.getenv('EXTRACTIVE_THRESHOLD', -7.0))
ANSWERABILITY_JUDGE_MODEL = os.getenv('ANSWERABILITY_JUDGE_MODEL', 'google/flan-t5-large')
EXTRACTIVE_TOP_CHUNKS = int(os.getenv('EXTRACTIVE_TOP_CHUNKS', 2))
# flan-t5's encoder truncates at 512 tokens regardless of model size (base or
# large); leaves headroom for the instruction template + question wrapped
# around the retrieved-chunk context.
MAX_INPUT_TOKENS = int(os.getenv('MAX_INPUT_TOKENS', 480))
# Passage budget for chat (decoder only) generator models such as Qwen, which
# read far more than FLAN-T5's 512 token encoder; 3000 fits all top 5
# passages. core/generation.py picks the model type from its config.
GENERATION_CONTEXT_TOKENS = int(os.getenv('GENERATION_CONTEXT_TOKENS', 3000))
# 'float32' is the safe CPU default; 'bfloat16' halves memory on CPUs that
# support it, usually at some speed cost.
GENERATOR_DTYPE = os.getenv('GENERATOR_DTYPE', 'float32')

# --- Agent orchestration / Gemini (routing, clarification, structural reflection only —
#     never receives raw clinical note text; see reflect_structure_node design notes) ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-3.6-flash')
# Off by default: this call is advisory-only (it never blocks or changes the
# response) and its result isn't rendered anywhere in the UI today, so it's
# pure latency once a Gemini key is configured. Set true to re-enable.
ENABLE_EXTERNAL_REFLECTION = _bool('ENABLE_EXTERNAL_REFLECTION', False)
# Gemini's API rejects a deadline below 10s outright, so this must stay >= 10.
GEMINI_TIMEOUT_SECONDS = int(os.getenv('GEMINI_TIMEOUT_SECONDS', 12))
MAX_AGENT_STEPS = int(os.getenv('MAX_AGENT_STEPS', 6))

# --- Tools ---
OPENFDA_TIMEOUT_SECONDS = int(os.getenv('OPENFDA_TIMEOUT_SECONDS', 8))

# --- Flask ---
FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'dev-insecure-change-me')
FLASK_DEBUG = _bool('FLASK_DEBUG', False)
FLASK_HOST = os.getenv('FLASK_HOST', '127.0.0.1')
FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
