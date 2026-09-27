<div align="center">

# 🏥 Clinical Evidence Assistant

**An agentic assistant that answers clinical questions from real hospital discharge notes, and traces every answer back to the passage it came from.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-Dashboard-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-6E56CF?style=for-the-badge)](https://faiss.ai/)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-1C3C3C?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![Deployment](https://img.shields.io/badge/Deployment-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://clinical-rag-mimic-ecvl.vercel.app/)
[![Dataset](https://img.shields.io/badge/Data-MIMIC--IV-0078D4?style=for-the-badge)](https://physionet.org/content/mimic-iv-note/)
[![Status](https://img.shields.io/badge/Status-Completed-2ea44f?style=for-the-badge)](.)

<br/>

*MIMIC-IV-Note v2.2 · 331,793 discharge notes · Section-aware retrieval · Local FLAN-T5 generation · Answers checked against their sources before display*

</div>

---

## 🎬 Live Demo

[![Open Live App](https://img.shields.io/badge/Open%20Live%20App%20%F0%9F%9A%80-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://clinical-rag-mimic-ecvl.vercel.app/)

Deployed on Vercel, no setup needed. Ask a clinical question, see the passages the answer was built from, and read the pipeline breakdown, EDA, and evaluation results in the System Overview tab.

> **What the live demo runs on.** The hosted app answers from **ten fabricated discharge notes**, never from MIMIC-IV-Note. The PhysioNet Data Use Agreement forbids sharing access to the restricted data, and a public site returning real note excerpts to anonymous visitors would do exactly that. It performs genuine retrieval-augmented generation over those fabricated notes through the Gemini API, with the same faithfulness check the local pipeline uses. See [Demo Mode](#-demo-mode-no-credentialed-access-required) for why, and [Public deployment](#public-deployment-vercel) for how.
>
> The **local** pipeline is the one that runs on real data, and it runs entirely on CPU with no data leaving the machine. Keep `GEMINI_API_KEY` blank for MIMIC-IV-Note runs: routing and clarification are the only steps that would call out, and they fall back to local keyword rules without a key.

---

## 🎯 Why This Exists

**The problem.** A discharge note can run to thousands of words, and the one line you need (a discharge dose, a potassium value, why a drug was stopped) sits in one section of it. A general chatbot will answer that question fluently, but it won't show you where the answer came from, and when the note is silent it tends to fill the gap with something plausible.

**Who it's for.** People who pull facts out of clinical notes and then have to check them: clinical informatics researchers, chart reviewers, and anyone building question-answering over health records. It doesn't diagnose or suggest treatment.

**What people use today.** Keyword search in the record finds the word but leaves you to read around it. Pasting notes into a hosted chatbot gives you an answer with no trail back to the source, and with MIMIC-IV the data use agreement rules that out anyway. A basic RAG pipeline shows the passages it used, but it still displays whatever the model wrote, supported or not.

**What this does differently.** The answer has to pass a check before you see it. Every draft is compared against the retrieved passages on content words and numbers, gets one retry, and turns into a refusal if it still doesn't hold up. Dosage questions skip the patient note entirely and go to the FDA label. The data agreement shaped the deployment too: real notes never leave the local machine, and the public demo runs on ten notes I wrote myself.

## 🔎 System Scope

Clinical Evidence Assistant answers questions over real de-identified discharge notes from the MIMIC-IV dataset. Ask a plain-language clinical question and it retrieves the relevant sections, routes to the right tool (record lookup or an FDA label lookup), generates an answer locally, and checks that answer against the retrieved text before showing it. If the notes don't contain the answer, it says so instead of guessing.

The default restricted-data mode works entirely on CPU with no paid APIs and no data leaving the local machine.

The interface opens on a case-study page covering what the system is, the two datasets behind it, the measured results for each, and what it deliberately does not decide. From there it hands over to a two-tab workspace where one tab is for asking questions and the other shows the full pipeline, EDA and evaluation results.

## 🧪 Evaluation

The original evaluation was a 10 question keyword check. One question moved the score by 10 points, so it could not separate a real change from noise, and it mixed retrieval and generation into one number. It is now a golden set of 140 questions across 8 question types on the fabricated demo notes, scored with retrieval metrics and generation metrics separately, with confidence intervals on every number. A set of about 100 questions on the real notes uses the same format and stays on my machine.

Everything lives in `evals/`, separate from `core/evaluation.py`, which stays as the original smoke test. Rules that hold for every run:

- Real data results go to `outputs/eval/` or a `*.local.*` file, both ignored by git. Only aggregate numbers from real runs appear here.
- Evaluation runs are offline. Gemini is switched off, routing uses the local keyword rules, and the openFDA lookup is replaced by a stub, since only the routing decision is scored. A real data run refuses to start while `GEMINI_API_KEY` is set.
- Relevance is defined by note and section header (for example admission 29000002, Discharge Medications), so a change to chunk size does not invalidate the gold labels.

### How the numbers were kept honest

The 140 questions come in two groups. The first 100 (`evals/golden/demo_questions.jsonl`) were used to find and tune improvements, split 50/50 into a dev half and a test half before any tuning (`evals/golden/demo_split.json`). The other 40 (`evals/golden/demo_holdout.jsonl`) were written and committed before the last round of changes and were scored exactly once afterwards. The headline numbers below come from those 40. I wrote every question and approved them in bulk without a line by line review; each record says so in its `notes` field.

### Headline results (40 question holdout)

| Measure | Before the changes | After, local agent | After, live public app (Gemini) |
|---|---|---|---|
| Retrieval recall@5 | 0.701 (0.540 to 0.851) | 0.908 (0.810 to 0.983) | 0.923 (0.845 to 0.982) |
| Retrieval MRR | 0.428 (0.286 to 0.561) | 0.865 (0.761 to 0.948) | not measured |
| Answer correctness | 34%, 10 of 29 (20% to 53%) | 69%, 20 of 29 (51% to 83%) | 61%, 17 of 28 (42% to 76%) |
| Refusal when the notes lack the answer | 17%, 1 of 6 (3% to 56%) | 83%, 5 of 6 (44% to 97%) | 100%, 6 of 6 (61% to 100%) |
| Routing to clarify or the FDA label | 100%, 5 of 5 | 100%, 5 of 5 (57% to 100%) | not scored |

Intervals are 95%: bootstrap over questions for retrieval, Wilson for rates. The paired difference in recall@5 between the new and the previous retrieval on the holdout is 0.207 (0.034 to 0.414), so the retrieval gain is real on questions that were never used for tuning. The "before" column is the code as it was before these changes (commit ad0262a), run on the same 40 questions. On them, answer correctness went from 34% to 69% and refusal from 1 in 6 to 5 in 6. 69% is short of the 80% I aimed for, and the intervals are wide because the holdout is small.

On the 100 original questions, which were used to choose the changes and therefore overstate them, the local agent now scores 61 of 70 on answer correctness (87%, 77% to 93%) and 14 of 16 on refusal (88%), against 36 of 70 (51%) and 4 of 16 (25%) before. The holdout above is the fairer measure.

### What changed

Each change was chosen on the dev half and checked on questions that took no part in choosing it.

1. **Contextual index text.** Each chunk is embedded with a short line naming its note's presenting problem and primary diagnosis in front of it (Anthropic, contextual retrieval, 2024; HiQA, arXiv:2402.01767). A medication list never names the condition it treats, so before this change a question about "the pneumonia patient's medications" often found the right patient but the wrong section. The stored and displayed chunk text is unchanged. `CONTEXTUAL_INDEX=false` switches it off.
2. **Stemmed header matching.** The header boost now compares word stems, so "discharged" matches "Discharge Medications". `RETRIEVAL_STEM_HEADERS=false` switches it off.
3. **Hybrid answering.** A two part question is split and each part retrieved on its own (decomposed prompting, Khot et al., ICLR 2023). A part whose top passage is a numbered list, such as a medication list, is answered with that list word for word, because FLAN-T5 tended to stop after the first drug. Other parts are answered by FLAN-T5. `ANSWER_MODE=generative` restores the original path.
4. **Refusing when the notes lack the answer.** A question is refused when a FLAN-T5-large yes/no check says no part can be answered from the passages and a local cross encoder gives the best evidence sentence a low score. The yes/no prompt states that a recorded absence such as "no chest pain" counts as an answer; without that line it refused most negation questions.

Retrieval, the switch to hybrid answering and the refusal rule were each tried against alternatives that lost: BM25, reciprocal rank fusion, a cross encoder reranker, answering with FLAN-T5-large, fewer passages in the prompt, answering only with extracted sentences, and a relevance threshold alone. `evals/tune_retrieval.py`, `evals/tune_hybrid.py` and `evals/tune_round2.py` reproduce those comparisons.

### Retrieval comparison (holdout)

| Variant | recall@1 | recall@5 | MRR | recall@5 minus dense, paired |
|---|---|---|---|---|
| `dense` (app default: contextual index, stemmed header boost) | 0.718 | 0.908 | 0.865 | baseline |
| `dense_legacy` (the previous behaviour) | 0.218 | 0.701 | 0.428 | -0.207 (-0.414 to -0.034) |
| `dense_no_boost` | 0.649 | 0.954 | 0.797 | 0.046 (0.000 to 0.103) |
| `bm25` | 0.287 | 0.322 | 0.322 | -0.586 (-0.776 to -0.379) |
| `hybrid_rrf` | 0.339 | 0.557 | 0.441 | -0.351 (-0.534 to -0.167) |
| `hybrid_rrf_rerank` (`cross-encoder/ms-marco-MiniLM-L-6-v2`, Apache 2.0) | 0.167 | 0.523 | 0.345 | -0.385 (-0.586 to -0.178) |

`dense_no_boost` had the highest recall@5, but its paired interval touches zero and its MRR is lower, so I treat it as tied with `dense`. Every retriever found the right note for every question (note hit@5 of 1.000); the differences are all about picking the right section. The demo corpus is 10 notes and 101 chunks, so these numbers say more about ranking sections within a note than about search across thousands of notes.

### Where it still fails

On the holdout, the local agent answered 1 of 5 negation questions correctly ("Did the patient still have a fever at discharge?"), and the answerability check still refuses some of them. Two part questions were right 3 times out of 5. The live Gemini app mixes up patients on some questions, because the public runtime keeps its own Gemini embedding retrieval and did not get the contextual index; it also loses some points to the keyword assertions, which do not accept a correct paraphrase such as "insulin glargine at a dose of 20 units".

### Faithfulness check versus human labels

`python -m evals.label_tool --corpus demo` shows the question, the retrieved passages and the first draft, and records my label: supported, partially supported or unsupported. It never shows the check's own decision. `python -m evals.faithfulness_agreement --corpus demo` then reports agreement, Cohen's kappa, the false pass rate and the false refusal rate, next to my own agreement with myself on a relabelled 20.

TODO: these numbers need my labels, which I have not done yet.

From the code, three gaps in the check are already clear: it ignores "no" and "not" as stopwords, compares numbers against every retrieved passage at once, and skips a sentence with no content words, such as a lone "31.". A synthetic stress set (`evals/stress.py`) measures each of these once the labels exist.

### Regression gate

`python -m evals.check_regression` runs the default retriever on the 70 reviewed retrieval questions and fails when recall@5 drops by more than 0.02 or MRR by more than 0.03 against `evals/baseline_demo.json` (currently recall@5 0.9155 and MRR 0.8643). One lost question moves recall@5 by about 0.014, so the gate tolerates one borderline question and fails at two. `.github/workflows/eval.yml` runs it on every pull request with CPU torch, no secrets and no call to Gemini. `python -m evals.check_regression --update-baseline` prints the old and new values and refuses to run in CI.

To record the gate failing on a throwaway pull request:

1. `git checkout -b throwaway/weaker-retrieval`
2. In `config.py`, change the default of `CONTEXTUAL_INDEX` to `False`, which rebuilds the demo index without the note context in CI.
3. Commit, push the branch and open a pull request against `main`.
4. Wait for the **Retrieval regression gate** job, then copy the FAIL lines from its log.
5. Close the pull request without merging and delete the branch.

### Poisoned note test

`python -m evals.run_injection_eval` builds a temporary index of the ten demo notes plus three fabricated notes from `demo/adversarial_notes.py`. Each of those notes hides an instruction to include the harmless string CANARY-7731. The public demo index never contains them. Nine questions target the three notes.

Measured on 27 September 2026 with the final system:

| Generator | Poisoned passage retrieved | Canary in the final answer, of exposed questions | Check passed the canary answer |
|---|---|---|---|
| FLAN-T5, local agent | 6 of 9 | 2 of 6, 33% (10% to 70%) | 2 of 2 |
| Gemini, the public demo's generator | 4 of 5 answered | 2 of 4, 50% (15% to 85%) | 2 of 2 |

Four of the nine Gemini calls failed on this run, most likely on the free tier's quota, so Gemini answered five questions. Before the answering changes the local agent leaked the canary on 1 of 6 exposed questions. The extra leak comes from quoting medication lists word for word: one poisoned note hides its instruction inside the list itself, and a verbatim quote passes it through.

The faithfulness check passed every answer that carried the canary. It compares an answer with the retrieved passages, and the canary is in the passage, so an echoed instruction looks supported. Nine questions is a small sample and the intervals are wide.

### Original smoke test

| Metric | Real dataset (MIMIC-IV-Note v2.2) | Synthetic demo |
|---|---|---|
| Original smoke test (10 questions, keyword match) | 70% (7/10); 60% (6/10) after the index change | 80% (8/10) |
| Mean latency | ~3.4s per question | ~4.6s per question |

These runs call retrieval and generation directly, so routing, the faithfulness check and the refusal never took part in them. The real dataset figure was 7 of 10 before the contextual index and 6 of 10 after it, rerun on 27 September 2026. One question is the whole difference, which is the noise this evaluation was built to get past, and none of these ten questions names a patient, so they do not test what the contextual index is for. Whether the change helps on real notes needs the real golden set. No note text, answer or passage from the real runs is in this repository.

### Limitations

- The demo corpus is 10 fabricated notes and 101 chunks, and the holdout has 40 questions, so every interval is wide.
- One person (me) wrote the questions and approved them in bulk, and will label the answers.
- Answer correctness still rests on keyword assertions, which reject some correct paraphrases.
- The real notes come from one institution, Beth Israel Deaconess Medical Center. Real dataset numbers are aggregates only, and reproducing them needs credentialed PhysioNet access.
- Routing is scored with the local keyword rules, since evaluation runs keep Gemini off.
- Hybrid answering is slower: each question part runs a yes/no check with FLAN-T5-large as well as FLAN-T5-base.

## 🗂️ Dataset

| Detail | Value |
|---|---|
| Name | MIMIC-IV Clinical Discharge Notes |
| Source | PhysioNet (credentialed access required) |
| Total Notes | 331,793 |
| Unique Patients | 145,914 |
| Local Corpus | Random sample across the full dataset, no condition filtering; all derived chunks, indices, and measurements stay in the ignored `outputs/` directory |

Access requires completing CITI training and signing a PhysioNet Data Use Agreement. The dataset is not included in this repository. Earlier versions of this pipeline embedded only a diabetes-filtered subset; it now draws a random sample from across all conditions so the assistant can answer general clinical questions, not just diabetes-related ones.

## 🧪 Demo Mode (no credentialed access required)

The DUA above is a real constraint on public deployment, not just on this repository: it requires "I will not share access to PhysioNet restricted data with anyone else," and a live app that returns excerpts from real discharge notes to anonymous visitors would do exactly that, regardless of how the underlying files are stored. So a public-facing deployment (a shared demo link, a portfolio project) needs a dataset that was never restricted in the first place.

`demo/notes.py` holds ten fully fabricated discharge notes. No real patient, no real admission, every vital, lab, and diagnosis was invented for this project, written in the same section-header format MIMIC-IV-Note uses, so the existing chunking/embedding/retrieval/generation pipeline treats them identically to real notes. `demo/build_index.py` runs them through that same pipeline into `outputs_demo/`, a separate, git-safe directory from the real `outputs/` (which must never be committed).

To reuse this if you fork or extend the project:

```bash
# Build the demo index once, or after editing demo/notes.py
python -m demo.build_index

# Point the app at it (in .env, or as platform environment variables on deploy)
OUTPUT_DIR=outputs_demo/
DATASET_LABEL=Synthetic demo (fabricated notes; no patient data; real MIMIC-IV-Note data requires credentialed PhysioNet access)

# Optional: measure the demo's own accuracy the same way the real one is measured
python -m demo.evaluate

# For the public Vercel deployment specifically (see below): also precompute
# Gemini embeddings for the same chunks, once or after editing the notes
python -m demo.build_gemini_embeddings
```

The existing "Dataset" field in the UI reads `DATASET_LABEL` directly, so a demo deployment always honestly discloses what it's running on, with no separate banner or UI change needed.

See the Evaluation section above for the demo's measured accuracy and latency alongside the real dataset's.

### Public deployment (Vercel)

Live at **[clinical-rag-mimic-ecvl.vercel.app](https://clinical-rag-mimic-ecvl.vercel.app/)**.

The live public demo deployed on Vercel doesn't load SentenceTransformers/FAISS/FLAN-T5 in-process the way the local pipeline does. `requirements.txt` (the public runtime's dependency list) is small deliberately, since those packages together pull in several hundred megabytes of model weights that exceed Vercel's serverless function size limit. But it still runs real retrieval-augmented generation, not a stand-in: `demo/runtime.py` gets the same two model calls (embed the question, generate the answer) from Google's Gemini API instead of loading the models locally, so the deployed function stays small while doing genuine RAG.

Concretely, `demo/runtime.py`:
- Embeds the incoming question via Gemini's `gemini-embedding-001` and matches it against `outputs_demo/gemini_chunk_embeddings.pkl`, precomputed embeddings for the exact same 101 chunks (`outputs_demo/chunks_data.pkl`) the local pipeline retrieves against, re-ranked with the same header-relevance boost `core/retrieval.py` uses for the real dataset.
- Generates the answer with Gemini (`gemini-flash-lite-latest`, a separate model from `GEMINI_MODEL`) from the retrieved passages, using the same system prompt as the local FLAN-T5 pipeline. This is deliberately not the same model the local app uses for routing/reflection: that model's free tier caps `generateContent` at only 20 requests/day, discovered by hitting that exact limit during development, which is nowhere near enough for a public demo answering anonymous visitors. `gemini-flash-lite-latest`'s free tier (roughly 1,000-1,500 requests/day) is sized for that instead, on a completely separate quota.
- Runs that generated answer through the same local faithfulness check (`agent/reflection.py`'s content-word/numeric overlap check) the local pipeline uses, with one retry if it fails, before ever showing it.
- Falls back to a fully offline, deterministic keyword-matching method if `GEMINI_API_KEY` isn't configured or any Gemini call fails for any reason (quota, network, timeout). The app stays functional either way, just cruder without a key, the same pattern `agent/llm.py` already uses for local routing.

`python -m demo.evaluate_public` runs the same 10-question keyword-hit check against this pipeline. Its result is reported as a raw count, not a percentage, and deliberately not placed in the accuracy table above: a bare "100%" sitting next to the real dataset's 70% and the local synthetic demo's 80% would misleadingly read as "the deployed demo beats the real research pipeline," when all three numbers are actually the same small 10-question smoke test, not a benchmark that scales to general reliability.

| Metric | Public Vercel demo (Gemini-backed RAG) |
|---|---|
| Canonical checks passed | 10/10 (not a benchmark; see note above) |
| Mean Latency | ~1.5s per question |

## 🧠 How It Works

**Offline, once:** each discharge note goes through 8-step cleaning and section-aware chunking, gets embedded (384-dim), and lands in a FAISS `IndexFlatIP` index.

**Per question, live:** the question is routed to the right tool, not just handed straight to a retriever. This is an agent graph (`agent/graph.py`, built on LangGraph), not a single retrieve-then-generate pass.

```mermaid
flowchart TD
    Q["User question"] --> Route

    Route{"Route\n(Gemini if configured,\notherwise local keyword rules)"}
    Route -->|record question| Retrieve["Retrieve top-5 chunks\n(FAISS + sentence embedding)"]
    Route -->|drug dosage question| Dosage["openFDA label lookup"]
    Route -->|too ambiguous| Clarify["Ask a clarifying question"]

    Retrieve --> Generate["Flan-T5 local generation\n(grounded in retrieved text)"]
    Dosage -->|drug found| Card["Render FDA label card directly\n(no paraphrase of dosage text)"]
    Dosage -->|drug not found| Generate

    Generate --> Reflect["Local faithfulness check\n(content-word + numeric overlap\nagainst retrieved text)"]
    Reflect -->|unsupported, first try| Generate
    Reflect -->|supported| Respond["Response + cited passages"]
    Reflect -->|still unsupported| Refuse["Refusal, cites passages for review"]

    Clarify --> Respond
    Card --> Respond
```

The local faithfulness check is what actually gates a refusal. It runs on every turn, entirely on-device, and gets one retry if the first draft doesn't hold up. An optional external Gemini call can additionally annotate structural quality (coherence, hedging) if `GEMINI_API_KEY` is set, but it's advisory only and off by default, so it never overrides the local check. Because routing context and generated drafts can contain clinical facts, external Gemini features are for fabricated or otherwise unrestricted data only; leave `GEMINI_API_KEY` blank for MIMIC-IV-Note runs.

## ⚙️ Pipeline Configuration

| Parameter | Value |
|---|---|
| Embedding Model | `all-MiniLM-L6-v2` (384 dimensions) |
| FAISS Index Type | `IndexFlatIP` (cosine similarity via L2 normalisation) |
| Chunking Strategy | Section-aware (16 MIMIC headers), hierarchically sub-chunked at 180 words so long sections don't exceed the embedding model's 256-token limit |
| Chunk Overlap | 40 words |
| Top-K Retrieval | 5 chunks per query, de-duplicated against near-identical overlapping passages, re-ranked by a header-relevance boost (a chunk whose own section header lexically matches the question's words is favored, see `core/retrieval.py`'s `_header_boost`) |
| Generation Model | `google/flan-t5-base` |
| Repetition Controls | `repetition_penalty=1.15`, `no_repeat_ngram_size=4`, which suppresses the model re-emitting the same line multiple times, tuned down from an initial 1.3/3 after that stronger setting was found to also strip dose details (e.g. "60 mg PO daily") from medication lists |
| Embedding Batch Size | 64 |

**System prompt used** (see `core/generation.py` for the exact current version, including the one-shot example that teaches the model to answer in prose rather than copying the source's own numbering):
```
You are a clinical assistant answering questions about hospital discharge notes
covering a range of clinical conditions.
Answer the question below using only the context provided.
Include every distinct item the context mentions that is relevant to the question.
Use exact medical terms, doses, and values from the context, but write the answer
in your own words as complete sentences -- never copy numbering or list markers
from the context, and never state the same fact twice.
Be specific and complete rather than brief.
If the answer is not in the context, say: I cannot find this information in the provided notes.
```

## 🧹 Preprocessing Pipeline

Each discharge note goes through 8 steps before chunking:

1. Remove MIMIC de-identification placeholders
2. Remove empty lines containing only underscores or whitespace
3. Lowercase all text
4. Normalise whitespace to single spaces
5. Filter to keep only letters, numbers and basic punctuation
6. Tokenise using NLTK
7. Remove standard and custom clinical stop words and short tokens, **except negation terms** (no, not, nor, without, denies, ...). Clinical NLP research on MIMIC notes ([Wu et al. via ResearchGate](https://www.researchgate.net/publication/384777146_An_Efficient_Text_Cleaning_Pipeline_for_Clinical_Text_for_Transformer_Encoder_Models); [assertion detection survey, arXiv:2503.17425](https://arxiv.org/html/2503.17425v1)) flags that blanket stopword removal erases the ~13% of clinical findings that are negated, flipping "no chest pain" and "chest pain" into the same bag of words
8. Lemmatise using WordNetLemmatizer with POS tags

> **Note:** this cleaned/lemmatised text is used for the EDA visualisations (word clouds, word-frequency plots) only. The RAG knowledge base itself is built from the section-aware chunker operating on the *raw* note text (see below), so it was never affected by stopword removal in the first place. The fix above corrects what the preprocessing EDA represents, not the retrieval corpus.

## 🗂️ Repository Structure

```text
clinical-rag-mimic/
├── app.py                  # Flask entrypoint + /ask; picks the local or public-demo runtime
├── train.py                # Runs the full real-data pipeline and saves all outputs
├── config.py               # Central config, reads all settings from .env
│
├── core/                   # Real-data pipeline (MIMIC-IV-Note). Imported lazily by app.py,
│   │                       # so the public function never pulls in torch or faiss.
│   ├── preprocessing.py    # 8-step cleaning pipeline
│   ├── chunking.py         # Section-aware chunking with fallback
│   ├── embeddings.py       # Embedding generation and FAISS index
│   ├── retrieval.py        # Chunk retrieval functions
│   ├── generation.py       # Flan-T5 loader, prompt builder, answer generation
│   ├── data.py             # Data loading, EDA, condition subsets
│   ├── evaluation.py       # 10-question evaluation, saves results and plots
│   └── viz_style.py        # Shared matplotlib/seaborn styling for all plots
│
├── demo/                   # Everything synthetic. Nothing here reads MIMIC-IV-Note or
│   │                       # outputs/, so it all runs without credentialed access.
│   ├── notes.py            # The ten fabricated discharge notes (see Demo Mode above)
│   ├── runtime.py          # Gemini-backed RAG for the public Vercel deployment
│   ├── build_index.py      # Builds outputs_demo/ from the fabricated notes
│   ├── build_gemini_embeddings.py # Precomputes Gemini embeddings for demo/runtime.py
│   ├── build_chart_inputs.py      # Builds the cached inputs regenerate_ui_charts.py reads
│   ├── evaluate.py         # Same keyword-hit evaluation as core/evaluation.py, on the demo index
│   └── evaluate_public.py  # Same evaluation, run against demo/runtime.py directly
│
├── agent/                  # LangGraph agent: routing, tools, reflection, Gemini calls
├── scripts/
│   └── regenerate_ui_charts.py # Regenerates ignored OUTPUT_DIR/ui_charts from local caches
│
├── outputs_demo/           # Demo FAISS index, chunk store, Gemini embeddings. Synthetic, safe to commit
├── static/                 # CSS, JS, fonts, images, and project-created brand assets
├── templates/              # Flask HTML templates (index.html holds the case study and the workspace)
├── tests/                  # pytest unit tests + Playwright browser tests
├── requirements.txt        # Lightweight public deployment dependencies (Flask, google-genai, numpy)
├── requirements-local.txt  # Full local RAG and research dependencies
├── vercel.json             # Public deployment config (function size/duration limits)
└── .env.local.example      # Copy to .env for the full local pipeline (.env is gitignored)
```

The split is deliberate: `core/` is the only package that touches restricted data, and `demo/` is the only one the public deployment imports. `app.py` and `train.py` stay at the repository root because they are entrypoints, and Vercel resolves the serverless function from `app.py` there.

> **Note:** the `mimic-iv-note-deidentified-free-text-clinical-notes-2.2/` dataset folder and the `outputs/` folder are not included in this repository. The dataset requires credentialed PhysioNet access. The `outputs/` folder containing the preprocessed sample, FAISS index and chunk data is generated locally when you run `python train.py`.

## ⚙️ Setup and Usage

```bash
# 1. Clone the repository
git clone https://github.com/abinashprasana/clinical-rag-mimic.git
cd clinical-rag-mimic

# 2. Install dependencies
pip install -r requirements-local.txt

# 3. Download the dataset
# Get the MIMIC-IV-Note package from PhysioNet (requires credentialed MIMIC-IV access)
# https://physionet.org/content/mimic-iv-note/
# Extract it so discharge.csv.gz ends up at:
# mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv.gz
# (relative to the repository root, see DATA_PATH in core/data.py)

# 4. Run the training pipeline
python train.py

# 5. (Optional, synthetic/unrestricted data only) Enable Gemini routing/clarification
cp .env.local.example .env
# Get a free key at https://aistudio.google.com/apikey, then set it in .env:
# GEMINI_API_KEY=your-key-here
# Leave GEMINI_API_KEY blank for MIMIC-IV-Note or any other restricted data.
# Routing context and optional structural reflection can contain derived facts.
# .env is gitignored, so never commit it, and never paste a real key into a
# commit, issue, or chat. The app runs fine with no key at all (it falls
# back to local keyword-based routing); a key only upgrades that path.
# On a hosting platform, set GEMINI_API_KEY as a platform environment
# variable/secret rather than shipping a .env file with the deployment.

# 6. Launch the Flask dashboard
python app.py
```

Then open `http://localhost:5000` in your browser.

## 🧪 Limitations and Future Work

The embedding model and generative model are both general-purpose and were not trained on clinical or biomedical text. It's tempting to assume a domain-specific model like Bio_ClinicalBERT would improve retrieval, but a 2024 benchmark of clinical semantic search ([Kanithi et al., arXiv:2401.01943](https://arxiv.org/html/2401.01943v2)) found the opposite for short-context retrieval: generalist sentence-transformer models beat clinical-specific ones (their top generalist model hit 84.0% exact-match vs. 64.4% for the best clinical model, ClinicalBERT), so `all-MiniLM-L6-v2` is a reasonable choice here, not a placeholder to be swapped out. A domain-specific generation model (e.g. BioGPT) is more likely to help than a domain-specific embedding model would. The local corpus size is configurable, but restricted-data-derived corpus measurements and artifacts are deliberately kept out of the public repository. Because retrieval returns the single most relevant note, definitional questions ("What is hypertension?") tend to surface that patient's specific diagnosis rather than a general definition. That's correct grounded behaviour for this design, but worth knowing if you extend the evaluation set. The keyword-based evaluation is rigid and may penalise correct answers that use different but valid medical vocabulary. The pipeline is built on a single institution dataset from MIMIC-IV and may not generalise well to discharge notes from other hospitals or healthcare systems.

## 📌 Dataset Source

**MIMIC-IV-Note v2.2 (PhysioNet)**
Johnson et al. (2023). Available at: https://physionet.org/content/mimic-iv-note/
Credentialed access required. Open Government Licence.

## 🙋 Author

**Abinash Prasana Selvanathan**

⭐ If you found this useful, feel free to star the repo.
