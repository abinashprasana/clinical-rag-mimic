Corpus demo, 100 questions. Rates with 95% Wilson intervals. Routing is offline keyword rules (Gemini disabled); openFDA is stubbed.

| measure | result |
|---|---|
| Answer correctness (answer questions) | 42/70 = 60% [48%, 71%] |
| Refusal accuracy (unanswerable questions) | 14/16 = 88% [64%, 97%] |
| Routing accuracy (clarify and FDA questions) | 12/14 = 86% [60%, 96%] |
|   of which clarify | 4/6 = 67% [30%, 90%] |
|   of which FDA label lookup | 8/8 = 100% [68%, 100%] |

| category | correct |
|---|---|
| ambiguous | 4/6 = 67% [30%, 90%] |
| discharge_followup | 5/6 = 83% [44%, 97%] |
| fda_dosage | 8/8 = 100% [68%, 100%] |
| medication_list | 7/16 = 44% [23%, 67%] |
| multi_section | 4/12 = 33% [14%, 61%] |
| negation | 6/12 = 50% [25%, 75%] |
| single_fact | 20/24 = 83% [64%, 93%] |
| unanswerable | 14/16 = 88% [64%, 97%] |

| faithfulness gate | count |
|---|---|
| drafts checked | 88 |
| passed on first check | 74 |
| retried | 14 |
| retry produced a different draft | 0 |
| answerable questions refused by the gate | 14 |
| outcome: clarified | 4 |
| outcome: fda_card | 8 |
| outcome: model_refusal | 16 |
| outcome: refused_by_gate | 14 |
| outcome: shown | 58 |
