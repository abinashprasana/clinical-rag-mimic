Corpus demo, 100 questions. Rates with 95% Wilson intervals. Routing is offline keyword rules (Gemini disabled); openFDA is stubbed.

| measure | result |
|---|---|
| Answer correctness (answer questions) | 55/70 = 79% [68%, 87%] |
| Refusal accuracy (unanswerable questions) | 15/16 = 94% [72%, 99%] |
| Routing accuracy (clarify and FDA questions) | 12/14 = 86% [60%, 96%] |
|   of which clarify | 4/6 = 67% [30%, 90%] |
|   of which FDA label lookup | 8/8 = 100% [68%, 100%] |

| category | correct |
|---|---|
| ambiguous | 4/6 = 67% [30%, 90%] |
| discharge_followup | 5/6 = 83% [44%, 97%] |
| fda_dosage | 8/8 = 100% [68%, 100%] |
| medication_list | 14/16 = 88% [64%, 97%] |
| multi_section | 8/12 = 67% [39%, 86%] |
| negation | 10/12 = 83% [55%, 95%] |
| single_fact | 18/24 = 75% [55%, 88%] |
| unanswerable | 15/16 = 94% [72%, 99%] |

| faithfulness gate | count |
|---|---|
| drafts checked | 88 |
| passed on first check | 85 |
| retried | 3 |
| retry produced a different draft | 0 |
| answerable questions refused by the gate | 3 |
| outcome: clarified | 4 |
| outcome: fda_card | 8 |
| outcome: model_refusal | 17 |
| outcome: refused_by_gate | 3 |
| outcome: shown | 68 |
