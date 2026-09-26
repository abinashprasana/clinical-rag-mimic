# Writing the real question set

The real golden set lives in `evals/golden/real_questions.local.jsonl`. The `.local.` in the name keeps it out of git. Never rename it, and never copy lines from it into a tracked file, an issue, a commit message or a chat.

## Setup

1. Build the real index once with `python train.py`, so `outputs/chunks_data.pkl` exists.
2. Copy the template: `cp evals/golden/real_questions.template.jsonl evals/golden/real_questions.local.jsonl`.
3. List the notes and the sections that made it into the index:

   ```bash
   python -m evals.list_sections --corpus real
   python -m evals.list_sections --corpus real --note 20000001
   ```

   This prints admission ids and section headers only. A section shorter than 20 words is dropped by the chunker, so it will not appear here and cannot be used as a relevant entry.

## One record per line

| Field | What to put there |
|---|---|
| `id` | `q001`, `q002` and so on, unique in the file |
| `category` | one of `single_fact`, `medication_list`, `multi_section`, `negation`, `unanswerable`, `fda_dosage`, `ambiguous`, `discharge_followup` |
| `question` | the question as a user would type it; name the patient by a detail from the note (age, main problem), since the index holds many notes |
| `expected_behavior` | `answer` for the five answerable categories, `refuse` for `unanswerable`, `fda_lookup` for `fda_dosage`, `clarify` for `ambiguous` |
| `relevant` | for `answer` questions, every `{"note_id": ..., "section": ...}` pair whose text answers the question; `note_id` is the admission id (hadm_id) as printed by `list_sections`; leave the list empty for the other behaviours |
| `must_contain` | short exact terms the answer has to include, copied from the note, such as a drug with its dose (`furosemide 40 mg`) or a lab value (`8.1`) |
| `must_not_contain` | terms that would make the answer wrong, such as an admission dose when the question asks for the discharge dose |
| `reviewed` | set to `true` once you have checked the record; only reviewed records are scored |
| `notes` | anything a second reader should know |

Matching lowercases both sides and collapses whitespace. Numbers and units must match exactly, so `40 mg` does not match `40mg`. Keep terms short so a correct answer phrased differently still passes. For negation questions, prefer the negated finding as written in the note (`denies chest pain`), and add the positive claim to `must_not_contain`.

## Target counts

The same as the demo set, about 100 questions: single fact 24, medication list 16, multi section 12, negation 12, unanswerable 16, FDA dosage 8, ambiguous 6, discharge instructions or follow up 6.

## Check the file

```bash
python -m evals.golden --corpus real
```

The validator prints question ids, counts and error codes only. It checks the schema, that each category uses the right behaviour, and that every relevant pair exists in the index. A warning that a `must_contain` term is missing from the gold passages usually means a typo in the term or the wrong section.

## Rules while running on real data

Leave `GEMINI_API_KEY` blank. The evaluation scripts refuse to run on the real corpus when it is set, and they stub the openFDA lookup, so no text leaves the machine. Results for the real corpus go to `outputs/eval/`. Only aggregate numbers from those runs may be copied into the README.
