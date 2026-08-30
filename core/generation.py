import re
import threading
import time
from transformers import pipeline
import config

# The Flask app serves multiple concurrent requests (waitress runs several
# worker threads), but a single shared transformers pipeline instance isn't
# safe to call .generate() on from two threads at once -- serialize access
# so concurrent requests queue instead of racing on the same model state.
_generator_lock = threading.Lock()

_EXAMPLE = (
    'Example:\n'
    'Context:\nDischarge Medications: 1. Atorvastatin 20 mg PO DAILY 2. Aspirin 81 mg PO DAILY '
    '3. Metoprolol 25 mg PO BID\n'
    'Question: What medications were prescribed at discharge?\n'
    'Answer: The patient was discharged on three medications: atorvastatin 20 mg by mouth daily, '
    'aspirin 81 mg by mouth daily, and metoprolol 25 mg by mouth twice daily.\n\n'
)


def _assemble(question, context):
    return (
        f'You are a clinical assistant answering questions about hospital discharge notes '
        f'covering a range of clinical conditions. '
        f'Answer the question below using only the context provided. '
        f'Include every distinct item the context mentions that is relevant to the question -- '
        f'do not stop after the first one or two if more are listed. '
        f'Use exact medical terms, doses, and values from the context in your answer, but write '
        f'the answer in your own words as complete sentences -- never copy numbering or list '
        f'markers from the context, and never state the same fact twice. '
        f'Be specific and complete rather than brief. '
        f'If the answer is not in the context, say: I cannot find this information in the provided notes.\n\n'
        f'{_EXAMPLE}'
        f'Now answer this one the same way, covering every relevant item.\n\n'
        f'Context:\n{context}\n\n'
        f'Question: {question}\n\n'
        f'Answer:'
    )

def build_prompt(question, chunks, tokenizer=None):
    chunk_texts = [c['chunk_text'] if isinstance(c, dict) else c for c in chunks]
    # Keep prompt labels compact; section-header relevance is handled once,
    # in retrieval.py's header-boost re-ranking, instead of duplicating that
    # signal as extra natural-language framing here.
    context = '\n\n'.join([f'[Chunk {i+1}]\n{c}' for i, c in enumerate(chunk_texts)])

    if tokenizer is None:
        return _assemble(question, context)

    # flan-t5's encoder silently truncates past 512 tokens regardless of
    # model size -- a 5-chunk x 180-word context runs well past that, so
    # without accounting for the instruction template's own overhead the
    # model was already being fed truncated/garbled input. Measure the
    # template's fixed cost first, then truncate only the context to fit
    # what's actually left of the budget.
    overhead = len(tokenizer.encode(_assemble(question, ''), add_special_tokens=False))
    context_budget = max(config.MAX_INPUT_TOKENS - overhead, 50)

    context_tokens = tokenizer.encode(context, add_special_tokens=False)
    if len(context_tokens) > context_budget:
        context = tokenizer.decode(context_tokens[:context_budget], skip_special_tokens=True)

    return _assemble(question, context)

def _clean_generated_text(text):
    """Defense-in-depth cleanup applied to every answer, not just list-shaped
    ones: normalizes whitespace and drops exact duplicate sentences that slip
    past decoding controls. Deliberately does not touch numbers/numbering --
    a stray-looking "10." could be a real clinical value, too risky to strip
    with a regex in a clinical answer. The "[Chunk N]" label IS always safe
    to strip -- it's build_prompt's own internal formatting, injected into
    the model's input, never legitimate clinical content -- and the model
    sometimes echoes it back verbatim instead of only the passage text."""
    text = re.sub(r'\[Chunk\s*\d+\]', '', text)
    collapsed = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', collapsed)
    seen = set()
    deduped = []
    for sentence in sentences:
        key = sentence.strip().lower()
        if key and key in seen:
            continue
        seen.add(key)
        deduped.append(sentence)
    return ' '.join(deduped).strip()


def generate_answer(question, retrieved_chunks, generator):
    start_time = time.time()
    # The fast (Rust-backed) tokenizer errors with "Already borrowed" if
    # called concurrently from two threads on the same instance -- so the
    # lock must cover build_prompt's tokenizer.encode() calls too, not just
    # the final generate call.
    with _generator_lock:
        prompt = build_prompt(question, retrieved_chunks, tokenizer=generator.tokenizer)
        result = generator(
            prompt,
            max_new_tokens=config.MAX_NEW_TOKENS,
            truncation=True,
            # 1.3/3 stopped exact duplicate lines but was aggressive enough to
            # also penalize the repeated "mg PO <frequency>" pattern across a
            # medication list, causing the model to drop dosing information
            # entirely to avoid the penalty. 1.15/4 verified to still block
            # the original duplicate-line bug while retaining full doses.
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
        )
    latency = time.time() - start_time
    return _clean_generated_text(result[0]['generated_text']), latency

def load_generator():
    print(f'Loading {config.LOCAL_GENERATOR_MODEL}...')
    generator = pipeline(
        'text2text-generation',
        model=config.LOCAL_GENERATOR_MODEL,
        max_new_tokens=config.MAX_NEW_TOKENS,
    )
    print('Local generator loaded.')
    return generator
