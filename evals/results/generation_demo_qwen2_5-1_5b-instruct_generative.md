Corpus demo, 100 questions. Rates with 95% Wilson intervals. Routing is offline keyword rules (Gemini disabled); openFDA is stubbed.

| measure | result |
|---|---|
| Answer correctness (answer questions) | 49/70 = 70% [58%, 79%] |
| Refusal accuracy (unanswerable questions) | 14/16 = 88% [64%, 97%] |
| Routing accuracy (clarify and FDA questions) | 12/14 = 86% [60%, 96%] |
|   of which clarify | 4/6 = 67% [30%, 90%] |
|   of which FDA label lookup | 8/8 = 100% [68%, 100%] |

| category | correct |
|---|---|
| ambiguous | 4/6 = 67% [30%, 90%] |
| discharge_followup | 5/6 = 83% [44%, 97%] |
| fda_dosage | 8/8 = 100% [68%, 100%] |
| medication_list | 10/16 = 62% [39%, 82%] |
| multi_section | 5/12 = 42% [19%, 68%] |
| negation | 10/12 = 83% [55%, 95%] |
| single_fact | 19/24 = 79% [60%, 91%] |
| unanswerable | 14/16 = 88% [64%, 97%] |

| faithfulness gate | count |
|---|---|
| drafts checked | 88 |
| passed on first check | 82 |
| retried | 6 |
| retry produced a different draft | 0 |
| answerable questions refused by the gate | 5 |
| outcome: clarified | 4 |
| outcome: fda_card | 8 |
| outcome: model_refusal | 13 |
| outcome: refused_by_gate | 6 |
| outcome: shown | 69 |
