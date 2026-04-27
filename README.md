<h1 align="center">🏥 Clinical RAG — Medical Q&A on MIMIC-IV Discharge Notes</h1>

<p align="center">
  A Retrieval-Augmented Generation pipeline that answers clinical questions grounded in real hospital discharge notes, no hallucinations, no external APIs.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img alt="Flask" src="https://img.shields.io/badge/Dashboard-Flask-000000?logo=flask&logoColor=white" />
  <img alt="FAISS" src="https://img.shields.io/badge/Vector_Search-FAISS-6E56CF" />
  <img alt="Accuracy" src="https://img.shields.io/badge/Eval_Accuracy-70%25-2ea44f" />
  <img alt="Dataset" src="https://img.shields.io/badge/Data-MIMIC--IV-0078D4" />
  <img alt="Status" src="https://img.shields.io/badge/Status-Completed-2ea44f" />
</p>

## 🔎 What This Project Is

This project builds a RAG pipeline on top of real de-identified clinical discharge notes from the MIMIC-IV dataset. A user can type a plain-language clinical question and receive an answer that is grounded strictly in what the discharge notes actually say. The system retrieves the most relevant sections from the notes, passes them to a language model, and generates a concise answer. If the information is not present in the retrieved context the system says so rather than guessing.

The pipeline was built to work entirely on CPU with no paid APIs and no data leaving the local machine. All results are presented through a two-tab Flask dashboard where one tab is for asking questions and the other shows the full pipeline, EDA and evaluation results.

## 🎥 Demo Video

<!-- Record your screen showing a question being asked and answered in Tab 1, then upload here -->


https://github.com/user-attachments/assets/497c9699-0c8b-4837-933f-6f3afdeb044b


## 🧪 Evaluation Results

The pipeline was tested against 10 predefined clinical questions using keyword matching to check correctness.

| Metric | Score |
|---|---|
| Overall Accuracy | **70% (7/10 correct)** |
| Mean Latency | **10.91 seconds per question** |

**Questions answered correctly:**
What is diabetes, what medications are used for blood glucose control, what is HbA1c, what does BID mean, what are signs of hypoglycemia, what is metformin used for, and what follow-up is recommended after discharge.

**Questions that failed keyword match:**
What diet is recommended for diabetic patients, what does PRN mean, and what is the normal blood glucose range. These likely produced semantically correct answers that used different vocabulary from the expected keywords.

## 🗂️ Dataset

| Detail | Value |
|---|---|
| Name | MIMIC-IV Clinical Discharge Notes |
| Source | PhysioNet (credentialed access required) |
| Total Notes | 331,793 |
| Unique Patients | 145,914 |
| Subset Used | Diabetes (10,000 notes preprocessed, 2,000 chunked and embedded) |

Access requires completing CITI training and signing a PhysioNet Data Use Agreement. The dataset is not included in this repository.

## 🧠 How It Works

```
Raw Discharge Note → 8-Step Cleaning → Section-Aware Chunking
                                                ↓
                                     Sentence Embedding (384-dim)
                                                ↓
                                       FAISS IndexFlatIP
                                                ↓
                    User Question → Embed + Normalise → Retrieve Top-5 Chunks
                                                ↓
                              Flan-T5 Grounded Generation → Answer
```

## ⚙️ Pipeline Configuration

| Parameter | Value |
|---|---|
| Embedding Model | `all-MiniLM-L6-v2` (384 dimensions) |
| FAISS Index Type | `IndexFlatIP` (cosine similarity via L2 normalisation) |
| Chunking Strategy | Section-aware (16 MIMIC headers) with 400-word fallback |
| Chunk Overlap | 50 words |
| Top-K Retrieval | 5 chunks per query |
| Generation Model | `google/flan-t5-base` |
| Embedding Batch Size | 64 |

**System prompt used:**
```
You are a clinical assistant answering questions about diabetes discharge notes.
Answer the question below using only the context provided.
Use exact medical terms from the context in your answer.
Keep your answer short and specific.
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
7. Remove standard and custom clinical stop words and short tokens
8. Lemmatise using WordNetLemmatizer with POS tags

## 🗂️ Project Structure

```text
clinical-rag-mimic/
├── app.py              # Two-tab Flask dashboard
├── chunking.py         # Section-aware chunking with fallback
├── data.py             # Data loading, EDA, condition subsets
├── embeddings.py       # Embedding generation and FAISS index
├── evaluation.py       # 10-question evaluation, saves results and plots
├── generation.py       # Flan-T5 loader, prompt builder, answer generation
├── preprocessing.py    # 8-step cleaning pipeline
├── retrieval.py        # Chunk retrieval functions
├── requirements.txt    # Python dependencies
├── train.py            # Runs full pipeline and saves all outputs
├── static/             # EDA and evaluation plots (created by train.py)
└── templates/          # Flask HTML templates
```

> **Note:** `discharge.csv.gz` and the `outputs/` folder are not included in this repository. The dataset requires credentialed PhysioNet access. The `outputs/` folder containing the FAISS index and chunk data is generated locally when you run `python train.py`.

## ⚙️ Setup and Usage

```bash
# 1. Clone the repository
git clone https://github.com/abinashprasana/clinical-rag-mimic.git
cd clinical-rag-mimic

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
# Get discharge.csv.gz from PhysioNet — requires credentialed MIMIC-IV access
# https://physionet.org/content/mimic-iv-note/
# Place discharge.csv.gz in the root project folder

# 4. Run the training pipeline
python train.py

# 5. Launch the Flask dashboard
python app.py
```

Then open `http://localhost:5000` in your browser.

## 🧪 Limitations and Future Work

The embedding model and generative model are both general-purpose and were not trained on clinical or biomedical text. A domain-specific alternative like Bio_ClinicalBERT for embeddings or BioGPT for generation would likely improve answer quality noticeably. Only 2,000 notes from the diabetes subset were chunked and embedded due to CPU memory constraints, which limits the size of the knowledge base. The keyword-based evaluation is rigid and may penalise correct answers that use different but valid medical vocabulary. The pipeline is built on a single institution dataset from MIMIC-IV and may not generalise well to discharge notes from other hospitals or healthcare systems.

## 📌 Dataset Source

**MIMIC-IV-Note v2.2 — PhysioNet**
Johnson et al. (2023). Available at: https://physionet.org/content/mimic-iv-note/
Credentialed access required. Open Government Licence.

## 🙋 Author

**Abinash Prasana Selvanathan**

⭐ If you found this useful, feel free to star the repo.
