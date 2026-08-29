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
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'outputs/')
# Shown in the UI's "Dataset" field. Set to something like "Synthetic demo
# notes (not real patient data)" when OUTPUT_DIR points at outputs_demo/ --
# see scripts/build_demo_index.py -- so a public deployment always honestly
# discloses when it's not running on real MIMIC-IV data.
DATASET_LABEL = os.getenv('DATASET_LABEL', 'MIMIC-IV-Note v2.2')

# --- Generation (local model, always stays on-device) ---
LOCAL_GENERATOR_MODEL = os.getenv('LOCAL_GENERATOR_MODEL', 'google/flan-t5-base')
MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', 200))
# flan-t5's encoder truncates at 512 tokens regardless of model size (base or
# large); leaves headroom for the instruction template + question wrapped
# around the retrieved-chunk context.
MAX_INPUT_TOKENS = int(os.getenv('MAX_INPUT_TOKENS', 480))

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
