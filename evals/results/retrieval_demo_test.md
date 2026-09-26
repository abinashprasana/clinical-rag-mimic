Corpus demo, 35 retrieval questions, top 5, mean with 95% bootstrap interval (1000 resamples, fixed seed).

The demo corpus is 10 notes and 101 chunks, so recall@5 saturates and differences between variants are small. Treat variants whose intervals overlap as tied on this corpus.

Overall
| variant | n | recall@1 | recall@3 | recall@5 | MRR | nDCG@5 | recall@5 minus dense (paired) | note hit@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | 35 | 0.393 [0.243, 0.550] | 0.564 [0.428, 0.714] | 0.721 [0.586, 0.843] | 0.582 [0.443, 0.724] | 0.589 [0.468, 0.717] | baseline | 1.000 | 20.4 | 35.0 |
| dense_contextual | 35 | 0.693 [0.557, 0.829] | 0.843 [0.729, 0.943] | 0.879 [0.793, 0.950] | 0.852 [0.752, 0.936] | 0.823 [0.730, 0.905] | 0.157 [0.014, 0.314] | 1.000 | 21.1 | 31.2 |

Category: discharge_followup
| variant | n | recall@1 | recall@3 | recall@5 | MRR | nDCG@5 | recall@5 minus dense (paired) | note hit@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | 3 | 0.333 [0.000, 1.000] | 0.333 [0.000, 1.000] | 0.333 [0.000, 1.000] | 0.333 [0.000, 1.000] | 0.333 [0.000, 1.000] | baseline | 1.000 | 18.0 | 20.6 |
| dense_contextual | 3 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.667 [0.000, 1.000] | 1.000 | 30.9 | 32.4 |

Category: medication_list
| variant | n | recall@1 | recall@3 | recall@5 | MRR | nDCG@5 | recall@5 minus dense (paired) | note hit@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | 8 | 0.375 [0.000, 0.628] | 0.375 [0.000, 0.628] | 0.625 [0.250, 0.875] | 0.438 [0.156, 0.720] | 0.483 [0.179, 0.750] | baseline | 1.000 | 25.0 | 32.9 |
| dense_contextual | 8 | 0.875 [0.625, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.938 [0.812, 1.000] | 0.954 [0.862, 1.000] | 0.375 [0.125, 0.750] | 1.000 | 18.5 | 22.5 |

Category: multi_section
| variant | n | recall@1 | recall@3 | recall@5 | MRR | nDCG@5 | recall@5 minus dense (paired) | note hit@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | 6 | 0.250 [0.083, 0.417] | 0.500 [0.250, 0.750] | 0.667 [0.500, 0.833] | 0.708 [0.458, 0.917] | 0.575 [0.407, 0.729] | baseline | 1.000 | 19.7 | 34.6 |
| dense_contextual | 6 | 0.333 [0.083, 0.500] | 0.500 [0.250, 0.750] | 0.500 [0.250, 0.750] | 0.750 [0.417, 1.000] | 0.524 [0.282, 0.716] | -0.167 [-0.417, 0.167] | 1.000 | 24.4 | 30.4 |

Category: negation
| variant | n | recall@1 | recall@3 | recall@5 | MRR | nDCG@5 | recall@5 minus dense (paired) | note hit@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | 6 | 0.542 [0.208, 1.000] | 0.792 [0.458, 1.000] | 0.792 [0.458, 1.000] | 0.722 [0.389, 1.000] | 0.722 [0.417, 1.000] | baseline | 1.000 | 20.3 | 23.7 |
| dense_contextual | 6 | 0.708 [0.375, 1.000] | 0.917 [0.750, 1.000] | 0.958 [0.875, 1.000] | 0.917 [0.750, 1.000] | 0.897 [0.774, 1.000] | 0.167 [0.000, 0.500] | 1.000 | 23.5 | 28.2 |

Category: single_fact
| variant | n | recall@1 | recall@3 | recall@5 | MRR | nDCG@5 | recall@5 minus dense (paired) | note hit@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | 12 | 0.417 [0.167, 0.669] | 0.667 [0.417, 0.876] | 0.875 [0.667, 1.000] | 0.607 [0.392, 0.812] | 0.663 [0.473, 0.844] | baseline | 1.000 | 25.0 | 35.4 |
| dense_contextual | 12 | 0.667 [0.333, 0.917] | 0.833 [0.583, 1.000] | 0.917 [0.792, 1.000] | 0.778 [0.583, 0.944] | 0.805 [0.632, 0.958] | 0.042 [-0.125, 0.250] | 1.000 | 19.9 | 24.4 |
