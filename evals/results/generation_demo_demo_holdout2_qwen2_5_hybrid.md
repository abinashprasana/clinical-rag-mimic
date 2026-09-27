Corpus demo, 40 questions. Rates with 95% Wilson intervals. Routing is offline keyword rules (Gemini disabled); openFDA is stubbed.

| measure | result |
|---|---|
| Answer correctness (answer questions) | 22/29 = 76% [58%, 88%] |
| Refusal accuracy (unanswerable questions) | 5/6 = 83% [44%, 97%] |
| Routing accuracy (clarify and FDA questions) | 5/5 = 100% [57%, 100%] |
|   of which clarify | 2/2 = 100% [34%, 100%] |
|   of which FDA label lookup | 3/3 = 100% [44%, 100%] |

| category | correct |
|---|---|
| ambiguous | 2/2 = 100% [34%, 100%] |
| discharge_followup | 2/3 = 67% [21%, 94%] |
| fda_dosage | 3/3 = 100% [44%, 100%] |
| medication_list | 6/6 = 100% [61%, 100%] |
| multi_section | 3/5 = 60% [23%, 88%] |
| negation | 2/5 = 40% [12%, 77%] |
| single_fact | 9/10 = 90% [60%, 98%] |
| unanswerable | 5/6 = 83% [44%, 97%] |

| faithfulness gate | count |
|---|---|
| drafts checked | 35 |
| passed on first check | 34 |
| retried | 1 |
| retry produced a different draft | 0 |
| answerable questions refused by the gate | 1 |
| outcome: clarified | 2 |
| outcome: fda_card | 3 |
| outcome: model_refusal | 5 |
| outcome: refused_by_gate | 1 |
| outcome: shown | 29 |
