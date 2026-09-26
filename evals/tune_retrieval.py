"""Dev split sweep over the three retrieval options, on the demo set only.

    python -m evals.tune_retrieval

Scores every on/off combination of contextual index text, stemmed header
matching and note-first ranking on the frozen dev half. The test half is
never touched here, so the option picked from this table can be scored on
test once without having been tuned on it.
"""
import itertools
import sys

from evals import common
from evals.corpus import load_corpus
from evals.golden import GOLDEN_FILES, load_questions, select
from evals.metrics import bootstrap_ci
from evals.retrievers import DenseRetriever, contextual_index
from evals.run_retrieval_eval import evaluate_variant


def main():
    import faiss
    from sentence_transformers import SentenceTransformer

    import config
    corpus = load_corpus('demo')
    questions = [q for q in select(load_questions(GOLDEN_FILES['demo']), split='dev')
                 if q['expected_behavior'] == 'answer']
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    indexes = {False: faiss.read_index(f'{corpus.output_dir}/faiss_index.index'),
               True: contextual_index(model, corpus)}
    rows = []
    for ctx, stem, note_first in itertools.product((False, True), repeat=3):
        retriever = DenseRetriever(model, indexes[ctx], corpus.chunks, corpus.provenance,
                                   stem_headers=stem, note_first=note_first)
        records = evaluate_variant(retriever, corpus, questions)
        cells = [common.fmt_ci(*bootstrap_ci([r[m] for r in records])) for m in ('recall@1', 'recall@5', 'mrr')]
        rows.append(['yes' if ctx else 'no', 'yes' if stem else 'no', 'yes' if note_first else 'no', *cells])
    print(f'Dev split, {len(questions)} retrieval questions, dense retriever with header boost.')
    print(common.text_table(['contextual', 'stemmed headers', 'note first', 'recall@1', 'recall@5', 'MRR'], rows))
    return 0


if __name__ == '__main__':
    sys.exit(main())
