Corpus demo, 100 questions. Rates with 95% Wilson intervals. Routing is offline keyword rules (Gemini disabled); openFDA is stubbed.

| measure | result |
|---|---|
| Answer correctness (answer questions) | 61/70 = 87% [77%, 93%] |
| Refusal accuracy (unanswerable questions) | 14/16 = 88% [64%, 97%] |
| Routing accuracy (clarify and FDA questions) | 12/14 = 86% [60%, 96%] |
|   of which clarify | 4/6 = 67% [30%, 90%] |
|   of which FDA label lookup | 8/8 = 100% [68%, 100%] |
| Original smoke test (10 questions, keyword match) | 8/10 = 80% [49%, 94%] |

| category | correct |
|---|---|
| ambiguous | 4/6 = 67% [30%, 90%] |
| discharge_followup | 6/6 = 100% [61%, 100%] |
| fda_dosage | 8/8 = 100% [68%, 100%] |
| medication_list | 13/16 = 81% [57%, 93%] |
| multi_section | 8/12 = 67% [39%, 86%] |
| negation | 12/12 = 100% [76%, 100%] |
| single_fact | 22/24 = 92% [74%, 98%] |
| unanswerable | 14/16 = 88% [64%, 97%] |

| faithfulness gate | count |
|---|---|
| drafts checked | 88 |
| passed on first check | 88 |
| retried | 0 |
| retry produced a different draft | 0 |
| answerable questions refused by the gate | 0 |
| outcome: clarified | 4 |
| outcome: fda_card | 8 |
| outcome: model_refusal | 17 |
| outcome: shown | 71 |

Original smoke test source: rerun now with demo/evaluate.py questions and keywords.
