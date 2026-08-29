import os
import re
from collections import Counter
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import viz_style
import config

MIMIC_SECTIONS = config.MIMIC_SECTIONS

# all-MiniLM-L6-v2 truncates at 256 tokens (~180-200 words for clinical text).
# Sections like "Brief Hospital Course" routinely run past that, so any chunk
# above this size gets recursively split rather than silently truncated at
# embedding time, following the hierarchical (section-then-length) chunking
# pattern used in clinical RAG literature (e.g. CLI-RAG, arXiv:2507.06715).
MAX_CHUNK_WORDS = config.MAX_CHUNK_WORDS
CHUNK_OVERLAP = config.CHUNK_OVERLAP
# Keep very short, low-information sections out of the retrieval index while
# preserving substantive one-line clinical sections.
MIN_CHUNK_WORDS = 20

def fixed_chunk(text, size=MAX_CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = ' '.join(words[start:start + size])
        if len(chunk.split()) >= MIN_CHUNK_WORDS:
            chunks.append(chunk)
        start += size - overlap
    return chunks

def _split_section(header, content, size=MAX_CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = content.split()
    if len(words) <= size:
        return [f'[{header}] {content}']
    sub_chunks = []
    start = 0
    while start < len(words):
        piece = ' '.join(words[start:start + size])
        if len(piece.split()) >= MIN_CHUNK_WORDS:
            sub_chunks.append(f'[{header}] {piece}')
        start += size - overlap
    return sub_chunks

def section_chunk(text):
    pattern = '|'.join([re.escape(h) for h in MIMIC_SECTIONS])
    parts = re.split(f'({pattern}):', text, flags=re.IGNORECASE)

    sections_found = [p.strip() for p in parts if any(
        p.strip().lower() == h.lower() for h in MIMIC_SECTIONS)]

    if len(sections_found) < 2:
        return fixed_chunk(text)

    chunks = []
    i = 0
    while i < len(parts) - 1:
        header_candidate = parts[i].strip()
        is_header = any(header_candidate.lower() == h.lower() for h in MIMIC_SECTIONS)
        if is_header and i + 1 < len(parts):
            content = parts[i + 1].strip()
            if len(content.split()) >= MIN_CHUNK_WORDS:
                chunks.extend(_split_section(header_candidate, content))
            i += 2
        else:
            i += 1

    return chunks if chunks else fixed_chunk(text)

def chunk_all_notes(df, text_col='text', max_notes=2000):
    all_chunks, chunk_sources = [], []
    for idx, row in df.head(max_notes).iterrows():
        note_chunks = section_chunk(str(row[text_col]))
        all_chunks.extend(note_chunks)
        chunk_sources.extend([idx] * len(note_chunks))
    return all_chunks, chunk_sources

def plot_chunk_diagnostics(all_chunks, chunk_sources, static_dir='static'):
    os.makedirs(static_dir, exist_ok=True)
    viz_style.apply_style()
    chunk_lengths = [len(c.split()) for c in all_chunks]

    # MAX_CHUNK_WORDS (180) bounds the section CONTENT before the "[Header] "
    # prefix is added, so finished chunks legitimately land a few words above
    # it -- that is not truncation risk. The real danger line is the embedding
    # model's ~256-token limit; clinical text runs ~1.3 tokens/word, so stay
    # conservative and flag anything past ~230 words as actually at risk.
    SAFE_WORD_LIMIT = 230
    at_risk = sum(1 for l in chunk_lengths if l > SAFE_WORD_LIMIT)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(chunk_lengths, bins=60, color=viz_style.PALETTE[0], edgecolor=viz_style.BORDER, linewidth=0.4)
    ax.axvline(MAX_CHUNK_WORDS, color=viz_style.PALETTE[4], linestyle=':', linewidth=1.5,
               label=f'{MAX_CHUNK_WORDS}-word section-content split target')
    ax.axvline(SAFE_WORD_LIMIT, color=viz_style.PALETTE[3], linestyle='--', linewidth=1.5,
               label=f'~{SAFE_WORD_LIMIT}-word safe limit (256-token embedding cutoff)')
    ax.set_xlabel('Chunk length (words, including section-header prefix)')
    ax.set_ylabel('Number of chunks')
    ax.set_title(f'Chunk Length Distribution ({at_risk}/{len(chunk_lengths):,} chunks actually at truncation risk)')
    ax.legend()
    viz_style.save(f'{static_dir}/eda_chunk_lengths.png')

    chunks_per_note = Counter(chunk_sources)
    counts = list(chunks_per_note.values())

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(counts, bins=range(1, max(counts) + 2), color=viz_style.PALETTE[4], edgecolor=viz_style.BORDER, align='left')
    ax.axvline(np.mean(counts), color=viz_style.PALETTE[3], linestyle='--', linewidth=1.5,
               label=f'Mean: {np.mean(counts):.1f} chunks/note')
    ax.set_xlabel('Chunks per note')
    ax.set_ylabel('Number of notes')
    ax.set_title('Chunks Produced per Note')
    ax.legend()
    viz_style.save(f'{static_dir}/eda_chunks_per_note.png')

def main():
    df_working = pd.read_pickle('outputs/df_notes_pp.pkl')
    print(f'Chunking {len(df_working):,} notes sampled from across the full dataset...')
    all_chunks, chunk_sources = chunk_all_notes(df_working, text_col='text', max_notes=len(df_working))

    chunk_lengths = [len(c.split()) for c in all_chunks]
    print(f'Total chunks produced: {len(all_chunks):,}')
    print(f'Average chunk length:  {np.mean(chunk_lengths):.0f} words')

    plot_chunk_diagnostics(all_chunks, chunk_sources)
    print('Chunk diagnostic plots saved to static/eda_chunk_lengths.png and static/eda_chunks_per_note.png')

    with open('outputs/chunks_temp.pkl', 'wb') as f:
        pickle.dump({'chunks': all_chunks, 'sources': chunk_sources}, f)
    print('Chunks saved to outputs/chunks_temp.pkl')

if __name__ == '__main__':
    main()
