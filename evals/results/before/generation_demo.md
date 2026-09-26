Corpus demo, 100 questions. Rates with 95% Wilson intervals. Routing is offline keyword rules (Gemini disabled); openFDA is stubbed.

| measure | result |
|---|---|
| Answer correctness (answer questions) | 36/70 = 51% [40%, 63%] |
| Refusal accuracy (unanswerable questions) | 4/16 = 25% [10%, 49%] |
| Routing accuracy (clarify and FDA questions) | 12/14 = 86% [60%, 96%] |
|   of which clarify | 4/6 = 67% [30%, 90%] |
|   of which FDA label lookup | 8/8 = 100% [68%, 100%] |
| Original smoke test (10 questions, keyword match) | 8/10 = 80% [49%, 94%] |

| category | correct |
|---|---|
| ambiguous | 4/6 = 67% [30%, 90%] |
| discharge_followup | 4/6 = 67% [30%, 90%] |
| fda_dosage | 8/8 = 100% [68%, 100%] |
| medication_list | 7/16 = 44% [23%, 67%] |
| multi_section | 0/12 = 0% [0%, 24%] |
| negation | 9/12 = 75% [47%, 91%] |
| single_fact | 16/24 = 67% [47%, 82%] |
| unanswerable | 4/16 = 25% [10%, 49%] |

| faithfulness gate | count |
|---|---|
| drafts checked | 88 |
| passed on first check | 86 |
| retried | 2 |
| retry produced a different draft | 0 |
| answerable questions refused by the gate | 2 |
| outcome: clarified | 4 |
| outcome: fda_card | 8 |
| outcome: model_refusal | 5 |
| outcome: refused_by_gate | 2 |
| outcome: shown | 81 |

Original smoke test source: rerun now with demo/evaluate.py questions and keywords.
