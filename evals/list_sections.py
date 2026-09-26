"""Lists note ids and the section headers present in the local index, so gold
`relevant` entries can be filled in accurately. Prints identifiers and counts
only, never note text. For the real corpus, run this locally and do not paste
the output anywhere tracked.

    python -m evals.list_sections --corpus real
    python -m evals.list_sections --corpus real --note 20000001
"""
import argparse

from evals.corpus import load_corpus


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--corpus', choices=['demo', 'real'], default='demo')
    parser.add_argument('--note', help='show only this note id (hadm_id)')
    args = parser.parse_args(argv)

    corpus = load_corpus(args.corpus)
    by_note = corpus.sections_by_note()
    note_ids = [args.note] if args.note else sorted(by_note)
    for note_id in note_ids:
        sections = by_note.get(note_id)
        if not sections:
            print(f'{note_id}: not in the {args.corpus} index')
            continue
        listed = ', '.join(
            f'{s or "(no header)"} x{n}' if n > 1 else (s or '(no header)')
            for s, n in sorted(sections.items(), key=lambda kv: kv[0] or '')
        )
        print(f'{note_id}: {listed}')
    print(f'\n{len(by_note)} notes, {len(corpus.chunks)} chunks in the {args.corpus} index.')


if __name__ == '__main__':
    main()
