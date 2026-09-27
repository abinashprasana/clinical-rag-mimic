Corpus demo, 100 questions. Rates with 95% Wilson intervals. Routing is offline keyword rules (Gemini disabled); openFDA is stubbed.

| measure | result |
|---|---|
| Answer correctness (answer questions) | 54/70 = 77% [66%, 85%] |
| Refusal accuracy (unanswerable questions) | 15/16 = 94% [72%, 99%] |
| Routing accuracy (clarify and FDA questions) | 12/14 = 86% [60%, 96%] |
|   of which clarify | 4/6 = 67% [30%, 90%] |
|   of which FDA label lookup | 8/8 = 100% [68%, 100%] |

| category | correct |
|---|---|
| ambiguous | 4/6 = 67% [30%, 90%] |
| discharge_followup | 5/6 = 83% [44%, 97%] |
| fda_dosage | 8/8 = 100% [68%, 100%] |
| medication_list | 15/16 = 94% [72%, 99%] |
| multi_section | 9/12 = 75% [47%, 91%] |
| negation | 5/12 = 42% [19%, 68%] |
| single_fact | 20/24 = 83% [64%, 93%] |
| unanswerable | 15/16 = 94% [72%, 99%] |

| faithfulness gate | count |
|---|---|
| drafts checked | 88 |
| passed on first check | 82 |
| retried | 6 |
| retry produced a different draft | 0 |
| answerable questions refused by the gate | 6 |
| outcome: clarified | 4 |
| outcome: fda_card | 8 |
| outcome: model_refusal | 19 |
| outcome: refused_by_gate | 6 |
| outcome: shown | 63 |
