Corpus demo, 40 questions. Rates with 95% Wilson intervals. Routing is offline keyword rules (Gemini disabled); openFDA is stubbed.

| measure | result |
|---|---|
| Answer correctness (answer questions) | 16/29 = 55% [38%, 72%] |
| Refusal accuracy (unanswerable questions) | 6/6 = 100% [61%, 100%] |
| Routing accuracy (clarify and FDA questions) | 5/5 = 100% [57%, 100%] |
|   of which clarify | 2/2 = 100% [34%, 100%] |
|   of which FDA label lookup | 3/3 = 100% [44%, 100%] |

| category | correct |
|---|---|
| ambiguous | 2/2 = 100% [34%, 100%] |
| discharge_followup | 2/3 = 67% [21%, 94%] |
| fda_dosage | 3/3 = 100% [44%, 100%] |
| medication_list | 5/6 = 83% [44%, 97%] |
| multi_section | 2/5 = 40% [12%, 77%] |
| negation | 0/5 = 0% [0%, 43%] |
| single_fact | 7/10 = 70% [40%, 89%] |
| unanswerable | 6/6 = 100% [61%, 100%] |

| faithfulness gate | count |
|---|---|
| drafts checked | 35 |
| passed on first check | 35 |
| retried | 0 |
| retry produced a different draft | 0 |
| answerable questions refused by the gate | 0 |
| outcome: clarified | 2 |
| outcome: fda_card | 3 |
| outcome: model_refusal | 7 |
| outcome: shown | 28 |
