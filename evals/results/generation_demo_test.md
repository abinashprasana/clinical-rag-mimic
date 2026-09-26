Corpus demo, 50 questions. Rates with 95% Wilson intervals. Routing is offline keyword rules (Gemini disabled); openFDA is stubbed.

| measure | result |
|---|---|
| Answer correctness (answer questions) | 22/35 = 63% [46%, 77%] |
| Refusal accuracy (unanswerable questions) | 8/8 = 100% [68%, 100%] |
| Routing accuracy (clarify and FDA questions) | 6/7 = 86% [49%, 97%] |
|   of which clarify | 2/3 = 67% [21%, 94%] |
|   of which FDA label lookup | 4/4 = 100% [51%, 100%] |

| category | correct |
|---|---|
| ambiguous | 2/3 = 67% [21%, 94%] |
| discharge_followup | 1/3 = 33% [6%, 79%] |
| fda_dosage | 4/4 = 100% [51%, 100%] |
| medication_list | 5/8 = 62% [31%, 86%] |
| multi_section | 2/6 = 33% [10%, 70%] |
| negation | 2/6 = 33% [10%, 70%] |
| single_fact | 12/12 = 100% [76%, 100%] |
| unanswerable | 8/8 = 100% [68%, 100%] |

| faithfulness gate | count |
|---|---|
| drafts checked | 44 |
| passed on first check | 44 |
| retried | 0 |
| retry produced a different draft | 0 |
| answerable questions refused by the gate | 0 |
| outcome: clarified | 2 |
| outcome: fda_card | 4 |
| outcome: model_refusal | 16 |
| outcome: shown | 28 |
