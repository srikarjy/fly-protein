# Query-budgeted search benchmark on ProteinGym (frozen)

Question: what does the fly-inspired representation/learner buy compared with established protein-search strategies under the same experimental query budget?

**Answer: under this protocol, nothing measurable.** The fly learner is statistically indistinguishable from a graph random walk and is at or below plain random search at every budget; regularized evolution, graph-adapted AdaLead, GP-BO and zero-shot ESM scoring all beat it at 50-200 queries.

## Protocol
- 10 ProteinGym assays from 10 unrelated proteins (single substitutions, 2.5-5k variants): BLAT (TEM-1), DYR, RASH, UBC9, NUD15, TPK1, MLAC, KKA2, TPMT, RNC. Candidate set = all variants of the assay.
- Budget = number of unique variants whose fitness is measured; re-reading an already-measured variant is free. Budgets reported: 50, 100, 200, 500 (every method is a prefix-consistent sequential procedure, so one 500-query run gives all budgets; checked by tests).
- Structural no-leakage rule: methods receive an `Oracle` (src/flyprotein/bench/oracle.py) and a `Static` object that has no fitness field. The only way to learn a fitness is `oracle.query(i)`. The evaluator alone reads the full landscape afterwards.
- Same shared start variant per (assay, seed) for all methods that use one; method-specific RNG streams; 20 seeds; deterministic logging (queries.npz; re-running a job reproduces the logs bit for bit; no duplicate queries; every run used exactly 500).
- Local methods (random walk, greedy, regularized evolution, graph-adapted AdaLead, fly) move on the same embedding kNN graph (k=15, ESM-2 8M embeddings, mean-centred cosine). Global methods (random search, GP-BO, zero-shot) may query any candidate.
- Metrics (per-assay percentile normalisation; raw DMS scores are never averaged across assays): best-percentile found vs queries, P(reach assay top 1%), P(top 0.1%), AUC of the best-percentile curve, per-assay tables. Unit of replication = assay; intervals are 95% bootstrap CIs over the 10 assays (seed means within assay). Paired differences are taken per assay.

## Methods and what each may see
| method | information used |
|---|---|
| random search | nothing |
| random walk (graph) | graph |
| greedy local search | graph + fitness of variants it queried (steepest ascent: queries every neighbour, moves to the best if strictly better, else random restart) |
| regularized evolution | graph + fitness of queried variants (population 20, tournament 5, child = unqueried graph neighbour of the winner) |
| GP-BO | PCA-50 of all candidate embeddings + fitness of queried variants; exact GP, Matern-5/2, EI, 10 initial queries (start + 9 random), hyperparameters from a fixed grid by marginal likelihood |
| graph-adapted AdaLead | as GP-BO's model; **adapted**, not the published code: mutation = random graph neighbour, no recombination, GP-mean surrogate, kappa=0.05, batch 10 (see src/flyprotein/bench/adalead.py) |
| fly learner / + memory / best tested | graph + FlyHash codes + the reward of the variants it moves to (fitness change scaled by the std of fitness values already queried; no global assay statistic). "Best tested" = replay 5, T=0.3, memory, the best config from the earlier sweep; no tuning was done for this benchmark |
| zero-shot ESM-2 | **uses no observed fitness measurements to construct its ranking**: a precomputed protein-language-model score for every candidate (a function of the sequence only), evaluated in descending score order with no feedback. A budget of B means the top-B ranked candidates are then measured against the assay; those B measurements are not used to train or adapt the zero-shot model. WT-marginal s = log p(mut) - log p(wt) at the position from one unmasked wild-type pass (Meier et al. 2021); masked-marginal masks the position. 8M/150M/650M WT-marginal, 8M/150M masked-marginal |

Zero-shot sanity (evaluation only): mean Spearman with measured fitness 0.20 (8M), 0.46 (150M), 0.51 (650M WT-marginal); masked-marginal 0.47 (150M).

## Headline results (mean over assays; [95% CI over assays])
P(reach assay top 1%), %:

| method | B=50 | B=100 | B=200 | B=500 |
|---|---|---|---|---|
| Random search | 40.0 [32.5, 47.5] | 65.0 [56.5, 72.5] | 88.5 [81.5, 94.0] | 100.0 |
| Greedy local search | 40.5 [31.0, 50.0] | 67.0 [56.0, 77.5] | 88.5 [85.0, 92.0] | 99.5 |
| Regularized evolution | 48.0 [43.5, 53.5] | 75.5 [69.5, 81.0] | 94.0 [88.5, 98.5] | 100.0 |
| GP-BO | 52.0 [44.0, 60.5] | 65.5 [56.0, 75.5] | 92.5 [87.0, 97.0] | 99.5 |
| graph-adapted AdaLead | 54.5 [47.0, 62.5] | 72.5 [64.5, 80.0] | 94.0 [91.0, 96.5] | 100.0 |
| ESM-2 650M zero-shot (WT) | 50.0 [20.0, 80.0] | 80.0 [50.0, 100.0] | 90.0 [70.0, 100.0] | 100.0 |
| **Fly learner** | 30.5 [22.0, 38.0] | 51.0 [37.5, 63.5] | 76.5 [65.5, 87.0] | 98.5 |
| Fly learner + memory | 31.5 [21.0, 42.5] | 54.5 [44.0, 65.5] | 78.5 [66.5, 88.5] | 98.0 |
| Fly best tested | 30.5 [21.5, 39.0] | 55.0 [43.0, 66.5] | 84.5 [73.0, 93.0] | 99.5 |

Paired difference vs the fly learner, P(top 1%) in points [95% CI]: graph-adapted AdaLead +24.0 [15.5, 32.0] / +21.5 [7.5, 36.5] / +17.5 [8.5, 26.5] at B=50/100/200; GP-BO +21.5 [15.0, 28.0] / +14.5 / +16.0; regularized evolution +17.5 [8.0, 27.0] / +24.5 / +17.5. Fly vs random search: -9.5 [-19.5, 0.5] / -14.0 [-28.5, 0.0] / -12.0 [-24.0, 0.0]. Graph random walk minus fly: -1.5 [-9.0, 6.5] / +3.0 [-5.0, 11.5] / +4.5 [-3.0, 12.5] (all CIs include 0, i.e. the fly is level with, or nominally below, a random walk on the same graph). Memory and the best tested configuration do not change this (fly + memory vs fly: +1.0 / +3.5 / +2.0, CIs include 0; best tested +8.0 [2.5, 14.0] at B=200 only).
Full tables (best percentile, top 1%, top 0.1%, AUC, paired differences vs random search / fly / greedy, per assay): tables.md. Plots: headline_curves.png, per_assay_curves.png, top1_by_budget.png.

## Reading the result
1. Random search is a strong baseline here: 50 random queries from about 3k variants already reach the 97.9th percentile on average, and by 500 queries every method is at 99.8+. Informative differences live at 50-200 queries; the 500-query column is saturated.
2. At 50-200 queries, model-based global/population methods (graph-adapted AdaLead, GP-BO, regularized evolution) and zero-shot ESM-2 lead; the fly learner does not improve on its own graph random walk, so the learned readout adds no measurable search signal in the query-budgeted setting. GP-BO uses the same ESM-2 8M embeddings (PCA) and gains +12 points over random search at B=50, so the embedding itself is exploitable; the fly learner's linear readout over random sparse codes does not exploit it under this budget.
3. Zero-shot ESM-2 builds its ranking without any observed fitness measurements (the query count only says how many top-ranked candidates were subsequently measured against the assay). It is competitive with, and at B=100 nominally above, the best search methods, but its assay-level intervals are wide (deterministic per assay, 10 assays) so it cannot be ranked against graph-adapted AdaLead/GP-BO with confidence. It is weak at pinpointing the very top (P(top 0.1%) is near 0 at B<=100 for most variants of it).
4. Cross-assay pretraining (results/transfer) remains a negative ablation: it produced weak but consistently positive held-out ranking signal (Spearman 0.04-0.14), yet failed to improve query-budgeted navigation and often harmed it. The evidence is that weak transferable ranking information is insufficient for effective sequential search under the current learner.

## Limitations (read before citing)
- 10 assays, one phenotype type mostly growth/stability; CIs over assays are wide for the deterministic zero-shot methods.
- Embedding-based methods (GP-BO, graph-adapted AdaLead, fly, graph) use ESM-2 8M embeddings (CPU-cheap, cached for all assays); zero-shot ESM-2 uses up to 650M. 650M masked-marginal scoring was not run (CPU-prohibitive; a paid GPU job was deliberately not launched). Larger-embedding versions of GP-BO/fly were not benchmarked.
- Graph-adapted AdaLead is not the published implementation. GP-BO, regularized evolution and graph-adapted AdaLead settings (10 initial queries, population 20, batch 10) were fixed a priori, not tuned; the fly methods were not tuned on these seeds, but the "best tested" fly configuration was chosen earlier on the TEM-1 landscape.
- Percentile ties: assays with coarse values (DYR has 145 distinct values) inflate percentiles slightly.
- The fly learner's reward is scaled online by the std of queried fitness values (x0.45, the regime it was set for); earlier experiments in this repo z-scored with global assay statistics, so absolute fly numbers differ from those experiments.
- The public web visualisation was not touched.
