"""Shared plumbing for the evaluation runners: where output may go, the
real-data guard, and plain text table formatting.

Output rules
* demo corpus, reviewed questions only  -> evals/results/ (tracked, aggregates only)
* demo corpus with unreviewed drafts     -> evals/runs/demo/ (gitignored, labelled UNREVIEWED)
* per question demo records             -> evals/runs/demo/ (gitignored)
* real corpus, everything               -> outputs/eval/ (gitignored); console gets aggregates only
"""
import datetime
import json
import math
import os
import subprocess
import sys

import config

RESULTS_DIR = os.path.join('evals', 'results')
DEMO_RUNS_DIR = os.path.join('evals', 'runs', 'demo')
REAL_DIR = os.path.join('outputs', 'eval')

UNREVIEWED_BANNER = (
    '*** UNREVIEWED: includes draft questions not yet marked reviewed. '
    'For development only; do not report these numbers. ***'
)
DEMO_SIZE_NOTE = (
    'The demo corpus is 10 notes and 101 chunks, so recall@5 saturates and '
    'differences between variants are small. Treat variants whose intervals '
    'overlap as tied on this corpus.'
)


def guard_real_data(corpus_name):
    """The real corpus may never reach an external API. Refuse to run when a
    Gemini key is configured, and blank it in-process as a second barrier."""
    if corpus_name != 'real':
        return
    if config.GEMINI_API_KEY:
        sys.exit('GEMINI_API_KEY is set. Leave it blank for real-data evaluation runs.')
    config.GEMINI_API_KEY = ''


def output_dirs(corpus_name, include_unreviewed):
    """(aggregate_dir, records_dir) for this run."""
    if corpus_name == 'real':
        return REAL_DIR, REAL_DIR
    if include_unreviewed:
        return DEMO_RUNS_DIR, DEMO_RUNS_DIR
    return RESULTS_DIR, DEMO_RUNS_DIR


def run_metadata(corpus_name, include_unreviewed, n_questions):
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unknown'
    return {
        'corpus': corpus_name,
        'reviewed_only': not include_unreviewed,
        'n_questions': n_questions,
        'run_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'git_commit': commit,
    }


def refresh_summary(corpus_name, include_unreviewed):
    """Rebuild the UI summary after a reviewed run; unreviewed runs never feed it."""
    if include_unreviewed:
        return
    from evals.ui_summary import DEMO_SUMMARY, REAL_SUMMARY, write_summary
    if corpus_name == 'real':
        write_summary(REAL_DIR, 'real', REAL_SUMMARY)
    else:
        write_summary(RESULTS_DIR, 'demo', DEMO_SUMMARY)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write('\n')


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)


def fmt_ci(mean, low, high, digits=3):
    if math.isnan(mean):
        return 'n/a'
    return f'{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]'


def markdown_table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|']
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(lines)


def text_table(headers, rows):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = '  '.join(str(h).ljust(w) for h, w in zip(headers, widths))
    out = [line, '  '.join('-' * w for w in widths)]
    out += ['  '.join(str(c).ljust(w) for c, w in zip(row, widths)) for row in rows]
    return '\n'.join(out)
