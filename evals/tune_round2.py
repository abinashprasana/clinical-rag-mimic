"""Round 2 sweep for hybrid answering on the 100 original demo questions.

    python -m evals.tune_round2

The 40 question holdout (evals/golden/demo_holdout.jsonl) is never read
here. Every signal is computed once per question part and cached, then each
rule variant is scored from the cache.
"""
import json
import os
import sys

from evals import common
from evals.golden import GOLDEN_FILES, contains, load_questions, select

CACHE = os.path.join('evals', 'runs', 'demo', 'tune_round2_cache.json')
REFUSAL = 'cannot find this information'


def compute(questions):
    import faiss
    from sentence_transformers import SentenceTransformer

    import config
    from core.extractive import best_unit, decompose, get_reranker, is_list_unit
    from core.generation import generate_answer, get_judge, is_answerable, load_generator
    from core.retrieval import retrieve_chunks
    from evals.corpus import load_corpus

    corpus = load_corpus('demo')
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    index = faiss.read_index(f'{corpus.output_dir}/faiss_index.index')
    reranker, generator, judge = get_reranker(), load_generator(), get_judge()
    cache = {}
    for q in questions:
        parts = []
        for part in decompose(q['question']):
            chunks = retrieve_chunks(part, model, index, corpus.chunks, corpus.provenance, k=config.DEFAULT_TOP_K)
            score, unit, chunk = best_unit(part, chunks, reranker)
            parts.append({
                'score': score, 'unit': unit, 'is_list': is_list_unit(unit),
                'unit_rank': [c['chunk_idx'] for c in chunks].index(chunk['chunk_idx']),
                'generated': generate_answer(part, chunks[:config.GENERATION_TOP_K], generator)[0],
                'answerable': is_answerable(part, chunks, judge),
            })
        cache[q['id']] = parts
    return cache


def add_top_passages(cache, questions):
    """Adds the top ranked passage body per part (retrieval only, no generation)."""
    import faiss
    from sentence_transformers import SentenceTransformer

    import config
    from core.extractive import _HEADER, decompose, is_list_unit
    from core.retrieval import retrieve_chunks
    from evals.corpus import load_corpus
    corpus = load_corpus('demo')
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    index = faiss.read_index(f'{corpus.output_dir}/faiss_index.index')
    for q in questions:
        for part, p in zip(decompose(q['question']), cache[q['id']]):
            top = retrieve_chunks(part, model, index, corpus.chunks, corpus.provenance, k=1)[0]
            p['top_body'] = _HEADER.sub('', top['chunk_text']).strip()
            p['top_is_list'] = is_list_unit(p['top_body'])


def answer_text(parts, variant):
    texts = []
    for p in parts:
        use_list = p['is_list'] and (variant < 2 or p['unit_rank'] == 0)
        text = p['unit'] if use_list else p['generated']
        if variant >= 4 and p['top_is_list']:
            text = p['top_body']   # the top ranked passage is a list: quote it whole
        if variant >= 1 and REFUSAL in text.lower() and p['answerable']:
            text = p['unit']   # judge says answerable, generator refused: fall back to the evidence
        texts.append(text)
    return ' '.join(dict.fromkeys(texts))


def refused(parts, threshold=0.0):
    if threshold is None:
        threshold = 0.0
    return not any(p['answerable'] for p in parts) and max(p['score'] for p in parts) < threshold


def main():
    questions = [q for q in select(load_questions(GOLDEN_FILES['demo']))
                 if q['expected_behavior'] in ('answer', 'refuse')]
    if os.path.exists(CACHE):
        with open(CACHE, encoding='utf-8') as f:
            cache = json.load(f)
    else:
        cache = compute(questions)
    if not all('top_is_list' in p for parts in cache.values() for p in parts):
        add_top_passages(cache, questions)
    with open(CACHE, 'w', encoding='utf-8') as f:
        json.dump(cache, f)

    labels = {0: ('round 1 rule', 0.0), 1: ('+ evidence fallback when the generator refuses', 0.0),
              2: ('+ lists only from the top ranked passage', 0.0),
              3: ('round 1 rule, refusal threshold -8', -8.0),
              4: ('threshold -8 + quote the top passage when it is a list', -8.0),
              5: ('threshold -6 + quote the top passage when it is a list', -6.0),
              6: ('threshold -10 + quote the top passage when it is a list', -10.0)}
    rows = []
    for variant, (label, threshold) in labels.items():
        ok_a, ok_r, generator_refusals = [], [], 0
        for q in questions:
            parts = cache[q['id']]
            text = answer_text(parts, variant)
            is_refused = refused(parts, threshold) or REFUSAL in text.lower()
            if q['expected_behavior'] == 'answer':
                generator_refusals += (not refused(parts, threshold)) and REFUSAL in text.lower()
                ok_a.append(not is_refused and all(contains(text, t) for t in q['must_contain'])
                            and not any(contains(text, t) for t in q['must_not_contain']))
            else:
                ok_r.append(is_refused)
        rows.append([label, f'{sum(ok_a)}/{len(ok_a)} = {sum(ok_a) / len(ok_a):.0%}',
                     f'{sum(ok_r)}/{len(ok_r)} = {sum(ok_r) / len(ok_r):.0%}', generator_refusals])
    print('All 100 original questions (dev and test halves); the holdout is not read.')
    print(common.text_table(['rule', 'answer correctness', 'refusal accuracy',
                             'answerable questions the generator refused'], rows))
    return 0


if __name__ == '__main__':
    sys.exit(main())
