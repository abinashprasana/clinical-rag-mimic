"""Loads a chunk store (demo or real) and maps every chunk to the
(note_id, section) pair the golden set uses for relevance.

note_id is the admission id (hadm_id) as a string, for both corpora: neither
index stores a MIMIC note_id, and MIMIC-IV-Note has one discharge note per
admission. The section comes from the "[Header]" prefix chunking.py writes;
chunks from the fixed-size fallback have no header and get section None, so
they can never count as relevant.
"""
import os
import pickle
import re
from collections import defaultdict

import config

CORPUS_DIRS = {'demo': 'outputs_demo', 'real': 'outputs'}
REAL_EVAL_DIR = os.path.join('outputs', 'eval')

_HEADER_RE = re.compile(r'^\[([^\]]+)\]')
_CANONICAL = {h.lower(): h for h in config.MIMIC_SECTIONS}


def canonical_section(header):
    if header is None:
        return None
    key = re.sub(r'\s+', ' ', header.strip().lower())
    key = key.replace('follow-up', 'followup').replace('follow up', 'followup')
    return _CANONICAL.get(key)


def chunk_section(chunk_text):
    match = _HEADER_RE.match(chunk_text)
    return canonical_section(match.group(1)) if match else None


class Corpus:
    def __init__(self, name, chunks, provenance, output_dir):
        self.name = name
        self.chunks = chunks
        self.provenance = provenance
        self.output_dir = output_dir
        self.keys = [
            (str(p['hadm_id']) if p.get('hadm_id') is not None else None, chunk_section(c))
            for c, p in zip(chunks, provenance)
        ]

    def sections_by_note(self):
        """{note_id: {section: chunk_count}}, ids and headers only."""
        out = defaultdict(lambda: defaultdict(int))
        for note_id, section in self.keys:
            if note_id is not None:
                out[note_id][section] += 1
        return out

    def has_entry(self, note_id, section):
        return (str(note_id), canonical_section(section)) in set(self.keys)

    def chunk_indices(self, note_id, section):
        target = (str(note_id), canonical_section(section))
        return [i for i, key in enumerate(self.keys) if key == target]


def load_corpus(name):
    if name not in CORPUS_DIRS:
        raise ValueError(f'unknown corpus {name!r}, expected one of {sorted(CORPUS_DIRS)}')
    output_dir = CORPUS_DIRS[name]
    with open(os.path.join(output_dir, 'chunks_data.pkl'), 'rb') as f:
        data = pickle.load(f)
    return Corpus(name, data['chunks'], data['provenance'], output_dir)
