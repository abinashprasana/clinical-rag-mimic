Corpus demo, 50 questions. Rates with 95% Wilson intervals. Routing is offline keyword rules (Gemini disabled); openFDA is stubbed.

| measure | result |
|---|---|
| Answer correctness (answer questions) | 30/35 = 86% [71%, 94%] |
| Refusal accuracy (unanswerable questions) | 8/8 = 100% [68%, 100%] |
| Routing accuracy (clarify and FDA questions) | 6/7 = 86% [49%, 97%] |
|   of which clarify | 2/3 = 67% [21%, 94%] |
|   of which FDA label lookup | 4/4 = 100% [51%, 100%] |

| category | correct |
|---|---|
| ambiguous | 2/3 = 67% [21%, 94%] |
| discharge_followup | 3/3 = 100% [44%, 100%] |
| fda_dosage | 4/4 = 100% [51%, 100%] |
| medication_list | 7/8 = 88% [53%, 98%] |
| multi_section | 4/6 = 67% [30%, 90%] |
| negation | 6/6 = 100% [61%, 100%] |
| single_fact | 10/12 = 83% [55%, 95%] |
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
| outcome: model_refusal | 9 |
| outcome: shown | 35 |
