"""Label tool, agreement statistics and the stress corruptions, on fabricated
records only."""
from evals.corpus import Corpus
from evals.faithfulness_agreement import agreement, self_agreement
from evals.label_tool import item_key, label_loop, pending, read_jsonl
from evals.stress import corrupt, run_stress

PASSAGE = '[Discharge Medications] Furosemide 60 mg PO daily. Potassium 20 mEq daily. Denies chest pain.'
CORPUS = Corpus('tiny', [PASSAGE], [{'subject_id': 1, 'hadm_id': 10}], output_dir=None)


def record(qid, draft, supported=True, outcome='shown'):
    return {'id': qid, 'category': 'single_fact', 'first_draft': draft, 'first_supported': supported,
            'outcome': outcome, 'passages': [0]}


def test_corruptions():
    out = corrupt('The patient takes furosemide 60 mg and denies chest pain.', PASSAGE)
    assert 'furosemide 187 mg' in out['number_swap_absent']
    assert 'furosemide 20 mg' in out['number_swap_in_context']
    assert 'bypass grafting' in out['insert_fact']
    assert 'denies' not in out['drop_negation']


def test_stress_counts_known_gate_blind_spots():
    counts = run_stress([('The patient takes furosemide 60 mg and denies chest pain.', [PASSAGE])])
    assert counts['number_swap_absent'] == {'n': 1, 'detected': 1}
    assert counts['insert_fact'] == {'n': 1, 'detected': 1}
    # 20 is in the passage, but as the potassium dose, not the furosemide one
    assert counts['number_swap_in_context'] == {'n': 1, 'detected': 1}
    # "chest pain" is only ever denied in the passage
    assert counts['drop_negation'] == {'n': 1, 'detected': 1}


def test_number_swaps_skip_list_numbering():
    passage = '[Discharge Medications] 1. Amlodipine 5 mg PO daily 2. Metformin 500 mg PO BID'
    out = corrupt('1. Amlodipine 5 mg PO daily', passage)
    assert out['number_swap_absent'].startswith('1. Amlodipine ')   # the marker is left alone
    assert out['number_swap_in_context'] == '1. Amlodipine 500 mg PO daily'


def test_agreement_statistics():
    records = [record('q001', 'a'), record('q002', 'b'), record('q003', 'c', supported=False,
                                                                  outcome='refused_by_gate'),
               record('q004', 'd')]
    labels = [
        {'key': item_key(records[0]), 'id': 'q001', 'label': 'supported', 'round': 1},
        {'key': item_key(records[1]), 'id': 'q002', 'label': 'unsupported', 'round': 1},
        {'key': item_key(records[2]), 'id': 'q003', 'label': 'supported', 'round': 1},
        {'key': item_key(records[3]), 'id': 'q004', 'label': 'partially_supported', 'round': 1},
        {'key': 'q009:stale', 'id': 'q009', 'label': 'supported', 'round': 1},
    ]
    result = agreement(records, labels)
    assert result['n'] == 4 and result['stale_labels'] == 1
    assert result['agreement']['k'] == 1          # only q001 agrees
    assert result['false_pass']['k'] == 2 and result['false_pass']['n'] == 2
    assert result['false_refusal']['k'] == 1 and result['false_refusal']['n'] == 2
    assert result['confusion']['not_supported']['supported'] == 2


def test_self_agreement():
    labels = [
        {'key': 'a', 'label': 'supported', 'round': 1}, {'key': 'a', 'label': 'supported', 'round': 2},
        {'key': 'b', 'label': 'unsupported', 'round': 1}, {'key': 'b', 'label': 'partially_supported', 'round': 2},
    ]
    result = self_agreement(labels)
    assert result['n'] == 2
    assert result['agreement_3way']['k'] == 1 and result['agreement_binary']['k'] == 2
    assert result['kappa_binary'] == 1.0


def test_label_loop_writes_resumable_labels(tmp_path):
    path = tmp_path / 'labels.jsonl'
    records = [record('q001', 'first draft'), record('q002', 'second draft'), record('q003', 'third')]
    answers = iter(['x', 's', 'k', 'q'])
    shown = []
    written = label_loop(pending(records, [], 1), {'q001': 'Q1', 'q002': 'Q2', 'q003': 'Q3'}, CORPUS,
                         str(path), 1, ask=lambda _: next(answers), out=shown.append)
    rows = read_jsonl(str(path))
    assert written == 1 and rows[0]['id'] == 'q001' and rows[0]['label'] == 'supported'
    assert 'first draft' not in path.read_text()        # labels never store answer text
    assert PASSAGE in shown and 'first draft' in shown
    assert [r['id'] for r in pending(records, rows, 1)] == ['q002', 'q003']


def test_relabel_picks_a_fixed_sample_of_first_round_items():
    records = [record(f'q{i:03d}', f'draft {i}') for i in range(1, 31)]
    labels = [{'key': item_key(r), 'id': r['id'], 'label': 'supported', 'round': 1} for r in records[:25]]
    first, second = pending(records, labels, 2, 20), pending(records, labels, 2, 20)
    assert [r['id'] for r in first] == [r['id'] for r in second]
    assert len(first) == 20 and {r['id'] for r in first} <= {r['id'] for r in records[:25]}
