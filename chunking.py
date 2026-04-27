import re
import numpy as np
import pandas as pd
import pickle

MIMIC_SECTIONS = [
    'Chief Complaint', 'History of Present Illness', 'Past Medical History',
    'Social History', 'Family History', 'Allergies', 'Physical Exam',
    'Pertinent Results', 'Brief Hospital Course', 'Medications on Admission',
    'Discharge Medications', 'Discharge Disposition', 'Discharge Diagnosis',
    'Discharge Condition', 'Discharge Instructions', 'Followup Instructions'
]

def fixed_chunk(text, size=400, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = ' '.join(words[start:start + size])
        if len(chunk.split()) >= 20:
            chunks.append(chunk)
        start += size - overlap
    return chunks

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
            if len(content.split()) >= 20:
                chunks.append(f'[{header_candidate}] {content}')
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

def main():
    print('Chunking first 2,000 diabetes notes...')
    df_working = pd.read_pickle('outputs/df_diabetes_pp.pkl').head(2000).copy()
    all_chunks, chunk_sources = chunk_all_notes(df_working, text_col='text', max_notes=2000)

    chunk_lengths = [len(c.split()) for c in all_chunks]
    print(f'Total chunks produced: {len(all_chunks):,}')
    print(f'Average chunk length:  {np.mean(chunk_lengths):.0f} words')
    
    with open('outputs/chunks_temp.pkl', 'wb') as f:
        pickle.dump({'chunks': all_chunks, 'sources': chunk_sources}, f)
    print('Chunks saved to outputs/chunks_temp.pkl')

if __name__ == '__main__':
    main()
