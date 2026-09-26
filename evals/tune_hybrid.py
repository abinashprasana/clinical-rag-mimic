"""Dev split sweep for hybrid answering (core/extractive.hybrid_parts).

    python -m evals.tune_hybrid

Each question is decomposed into parts and each part retrieved separately.
A part whose best evidence unit is a numbered list is answered with that list
verbatim; any other part is answered by FLAN-T5 from its own passages. The
question is refused when no part's best unit reaches the threshold. FLAN-T5
outputs are computed once and reused for every threshold. The test half is
never read here.
"""
import sys

from evals import common
from evals.golden import GOLDEN_FILES, contains, load_questions, select

THRESHOLDS = (-99.0, -8.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0)


def main():
    import faiss
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline

    import config
    from core.extractive import best_unit, decompose, get_reranker, is_list_unit
    from core.generation import generate_answer, is_answerable, load_generator
    from core.retrieval import retrieve_chunks
    from evals.corpus import load_corpus

    corpus = load_corpus('demo')
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    index = faiss.read_index(f'{corpus.output_dir}/faiss_index.index')
    reranker, generator = get_reranker(), load_generator()
    judges = {'flan-t5-base': generator,
              'flan-t5-large': pipeline('text2text-generation', model='google/flan-t5-large')}
    questions = [q for q in select(load_questions(GOLDEN_FILES['demo']), split='dev')
                 if q['expected_behavior'] in ('answer', 'refuse')]

    import os
    import pickle
    cache = os.path.join('evals', 'runs', 'demo', 'tune_hybrid_cache.pkl')
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            per_question = pickle.load(f)
        questions = [q for q in questions if q['id'] in per_question]
    else:
        per_question = compute(questions, corpus, model, index, reranker, generator, judges,
                               decompose, retrieve_chunks, best_unit, is_list_unit, generate_answer,
                               is_answerable, config)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, 'wb') as f:
            pickle.dump(per_question, f)
    report(questions, per_question, judges)
    return 0


def compute(questions, corpus, model, index, reranker, generator, judges, decompose, retrieve_chunks,
            best_unit, is_list_unit, generate_answer, is_answerable, config):
    per_question = {}
    for q in questions:
        parts = []
        for part in decompose(q['question']):
            chunks = retrieve_chunks(part, model, index, corpus.chunks, corpus.provenance, k=config.DEFAULT_TOP_K)
            score, unit, _ = best_unit(part, chunks, reranker)
            text = unit if is_list_unit(unit) else generate_answer(part, chunks, generator)[0]
            verdicts = {name: is_answerable(part, chunks, judge) for name, judge in judges.items()}
            parts.append((score, text, verdicts))
        per_question[q['id']] = parts
    return per_question


def report(questions, per_question, judges):
    deciders = [(f'score >= {t}', lambda parts, t=t: max(p[0] for p in parts) < t) for t in THRESHOLDS]
    deciders += [(f'{name} says no to every part', lambda parts, n=name: not any(p[2][n] for p in parts))
                 for name in judges]
    deciders += [(f'flan-t5-large no and score < {t}',
                  lambda parts, t=t: not any(p[2]['flan-t5-large'] for p in parts) and max(p[0] for p in parts) < t)
                 for t in (-2.0, 0.0, 2.0, 4.0)]
    rows = []
    for label, decide in deciders:
        answer_ok, refuse_ok, wrongly_refused = [], [], 0
        for q in questions:
            parts = per_question[q['id']]
            refused = decide(parts)
            text = ' '.join(p[1] for p in parts)
            if q['expected_behavior'] == 'answer':
                wrongly_refused += refused
                answer_ok.append(not refused and all(contains(text, t) for t in q['must_contain'])
                                 and not any(contains(text, t) for t in q['must_not_contain']))
            else:
                refuse_ok.append(refused or 'cannot find this information' in text.lower())
        rows.append([label, f'{sum(answer_ok)}/{len(answer_ok)} = {sum(answer_ok) / len(answer_ok):.0%}',
                     f'{sum(refuse_ok)}/{len(refuse_ok)} = {sum(refuse_ok) / len(refuse_ok):.0%}',
                     wrongly_refused])
    print('Dev split, hybrid answering (lists verbatim, other parts by FLAN-T5 per part).')
    print(common.text_table(['refusal rule', 'answer correctness', 'refusal accuracy',
                             'answerable questions refused'], rows))
    print()
    print('Answerable questions the flan-t5-large judge refuses (best evidence score per part):')
    for q in questions:
        parts = per_question[q['id']]
        if q['expected_behavior'] == 'answer' and not any(p[2]['flan-t5-large'] for p in parts):
            print(f"  {q['id']} {q['category']}: scores {[round(p[0], 1) for p in parts]}")


if __name__ == '__main__':
    sys.exit(main())
