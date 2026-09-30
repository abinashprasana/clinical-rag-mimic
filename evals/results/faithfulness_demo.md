No human labels match these generation records, so agreement with human labels is not reported (python -m evals.label_tool records them).

SYNTHETIC CORRUPTION (never blended with the human label numbers). Source answers: drafts the check passed (no human labels used).

| corruption | detected by the gate |
|---|---|
| number_swap_absent | 45/45 = 100% [92%, 100%] |
| number_swap_in_context | 35/45 = 78% [64%, 87%] |
| insert_fact | 71/88 = 81% [71%, 88%] |
| drop_negation | 18/18 = 100% [82%, 100%] |
