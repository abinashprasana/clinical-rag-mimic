# Holdout scored with the code before the improvements (commit ad0262a)

Run on 2026-09-27 from a temporary git worktree of ad0262a with evals/golden/demo_holdout.jsonl copied in. Console output of both runs, aggregates only.

```
Corpus demo, 29 retrieval questions, top 5, mean with 95% bootstrap interval (1000 resamples, fixed seed).

The demo corpus is 10 notes and 101 chunks, so recall@5 saturates and differences between variants are small. Treat variants whose intervals overlap as tied on this corpus.

Overall
variant  n   recall@1              recall@3              recall@5              MRR                   nDCG@5                recall@5 minus dense (paired)  note hit@5  p50 ms  p95 ms
-------  --  --------------------  --------------------  --------------------  --------------------  --------------------  -----------------------------  ----------  ------  ------
dense    29  0.218 [0.080, 0.379]  0.511 [0.322, 0.684]  0.701 [0.540, 0.851]  0.428 [0.286, 0.561]  0.480 [0.346, 0.603]  baseline                       1.000       14.3    20.0  

Category: discharge_followup

Corpus demo, 40 questions. Rates with 95% Wilson intervals. Routing is offline keyword rules (Gemini disabled); openFDA is stubbed.

measure                                       result                
--------------------------------------------  ----------------------
Answer correctness (answer questions)         10/29 = 34% [20%, 53%]
Refusal accuracy (unanswerable questions)     1/6 = 17% [3%, 56%]   
Routing accuracy (clarify and FDA questions)  5/5 = 100% [57%, 100%]
  of which clarify                            2/2 = 100% [34%, 100%]
  of which FDA label lookup                   3/3 = 100% [44%, 100%]

category            correct               
------------------  ----------------------
ambiguous           2/2 = 100% [34%, 100%]
discharge_followup  2/3 = 67% [21%, 94%]  
fda_dosage          3/3 = 100% [44%, 100%]
medication_list     1/6 = 17% [3%, 56%]   
multi_section       0/5 = 0% [0%, 43%]    
negation            1/5 = 20% [4%, 62%]   
single_fact         6/10 = 60% [31%, 83%] 
unanswerable        1/6 = 17% [3%, 56%]   

faithfulness gate                         count
----------------------------------------  -----
drafts checked                            34   
passed on first check                     32   
retried                                   2    
retry produced a different draft          0    
answerable questions refused by the gate  2    
outcome: clarified                        2    
outcome: fda_card                         4    
outcome: model_refusal                    4    
outcome: refused_by_gate                  2    
outcome: shown                            28   

Wrote evals\results/generation_demo.json and .md; per question records in evals\runs\demo/.
```
