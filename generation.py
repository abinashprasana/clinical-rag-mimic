import time
from transformers import pipeline

def build_prompt(question, chunks):
    context = '\n\n'.join([f'[Chunk {i+1}]\n{c}' for i, c in enumerate(chunks)])
    prompt = (
        f'You are a clinical assistant answering questions about diabetes discharge notes. '
        f'Answer the question below using only the context provided. '
        f'Use exact medical terms from the context in your answer. '
        f'Keep your answer short and specific. '
        f'If the answer is not in the context, say: I cannot find this information in the provided notes.\n\n'
        f'Context:\n{context}\n\n'
        f'Question: {question}\n\n'
        f'Answer:'
    )
    return prompt

def generate_answer(question, retrieved_chunks, generator):
    start_time = time.time()
    prompt = build_prompt(question, retrieved_chunks)
    result = generator(prompt, max_new_tokens=200)
    latency = time.time() - start_time
    return result[0]['generated_text'].strip(), latency

def load_generator():
    print('Loading google/flan-t5-base...')
    generator = pipeline('text2text-generation', model='google/flan-t5-base', max_new_tokens=200)
    print('Flan-T5 loaded.')
    return generator
