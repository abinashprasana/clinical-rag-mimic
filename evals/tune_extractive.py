"""Dev split sweep for extractive answering (core/extractive.py).

    python -m evals.tune_extractive

Runs decomposition, retrieval per part and cross encoder selection once on
the dev half, then scores answer correctness and refusal accuracy for a
range of refusal thresholds. The test half is never read here.
"""
import sys

from evals import common
from evals.golden import GOLDEN_FILES, contains, load_questions, select
from evals.metrics import wilson_ci

THRESHOLDS = (-8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)


def main():
    import faiss
    from sentence_transformers import SentenceTransformer

    import config
    from core.extractive import best_unit, decompose, get_reranker
    from core.retrieval import retrieve_chunks
    from evals.corpus import load_corpus

    corpus = load_corpus('demo')
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    index = faiss.read_index(f'{corpus.output_dir}/faiss_index.index')
    reranker = get_reranker()
    questions = [q for q in select(load_questions(GOLDEN_FILES['demo']), split='dev')
                 if q['expected_behavior'] in ('answer', 'refuse')]

    retrieved = {q['id']: [(part, retrieve_chunks(part, model, index, corpus.chunks, corpus.provenance,
                                                   k=config.DEFAULT_TOP_K))
                           for part in decompose(q['question'])] for q in questions}
    rows = []
    for top_chunks in (1, 2, 3):
      found = {qid: [best_unit(part, chunks, reranker, top_chunks) for part, chunks in parts]
               for qid, parts in retrieved.items()}
      for threshold in THRESHOLDS:
        answer_ok, refuse_ok = [], []
        for q in questions:
            units = []
            for hit in found[q['id']]:
                if hit and hit[0] >= threshold and hit[1] not in units:
                    units.append(hit[1])
            text = ' '.join(units)
            if q['expected_behavior'] == 'answer':
                answer_ok.append(bool(units) and all(contains(text, t) for t in q['must_contain'])
                                 and not any(contains(text, t) for t in q['must_not_contain']))
            else:
                refuse_ok.append(not units)
        a, r = wilson_ci(sum(answer_ok), len(answer_ok)), wilson_ci(sum(refuse_ok), len(refuse_ok))
        rows.append([top_chunks, threshold, f'{sum(answer_ok)}/{len(answer_ok)} = {a[0]:.0%}',
                     f'{sum(refuse_ok)}/{len(refuse_ok)} = {r[0]:.0%}'])
    print('Dev split, extractive answering, top unit per question part.')
    print(common.text_table(['sections', 'threshold', 'answer correctness', 'refusal accuracy'], rows))
    return 0


if __name__ == '__main__':
    sys.exit(main())
