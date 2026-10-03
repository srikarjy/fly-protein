# Cross-assay transfer of the fly learner (leave-one-assay-out)

Question: can KC->output weights learned on other ProteinGym landscapes make the 200-step search on an unseen landscape better than starting from zero?

## Design
- 10 assays from 10 different proteins (single mutants, 2.5-5k variants), ESM-2 8M embeddings computed on CPU (no paid GPU): TEM-1 (BLAT), DYR, RASH, UBC9, NUD15, TPK1, MLAC, KKA2, TPMT, RNC. Max pairwise 3-mer Jaccard of the sequences is 0.04. Other TEM-1 assays (Deng, Stiffler) are excluded.
- Each assay is held out in turn; pre-training uses only the other 9 (100 starts x 200 steps per assay, weights carried across assays), same FlyHash wiring (one per seed), same learner rule. Held-out fitness is never used before evaluation; only its embeddings (for centring), exactly as in the baseline.
- Evaluation: 50 starts x 200 steps, same starts for every arm; 10 seeds; metric = reach-top-1% gain over a random walk on the same graph and starts (points).
- Rewards are per-assay z-scored fitness times a constant (0.45 = TEM-1's raw std, the regime the learner was set for; 1.0 as a sensitivity run). This is a calibration assumption of the environment; reach rates themselves are rank-based.
- Controls: pre-training on shuffled fitness; frozen pretrained weights (no learning at test); prior shrunk by alpha (0.1 / 0.3), chosen post hoc, so only an optimistic upper bound.

## Results (gain over random walk, top 1%, pooled over 10 held-out assays x 10 seeds, ± SE)
| arm | scale 0.45 | scale 1.0 |
|---|---|---|
| chemotaxis (oracle) | +29.0 ± 1.3 | +18.7 ± 1.9 |
| zero start | -0.2 ± 0.9 | -5.6 ± 1.1 |
| zero start + memory in held-out | -1.8 ± 1.7 | -10.9 ± 2.0 |
| pretrained | **-13.1 ± 2.3** | -24.6 ± 1.9 |
| pretrained + memory | -7.1 ± 2.3 | -21.2 ± 2.1 |
| pretrained, frozen | -12.4 ± 2.3 | -25.5 ± 1.9 |
| pretrained on shuffled fitness | -15.2 ± 2.1 | -26.1 ± 1.9 |
| pretrained, alpha=0.1 (upper bound) | +0.5 ± 0.9 | -5.5 ± 1.2 |

Paired, scale 0.45: pretrained − zero = -12.9 ± 2.2 (3/10 assays better); pretrained+memory − zero+memory = -5.3 ± 2.2 (3/10); alpha=0.1 − zero = +0.7 ± 0.8 (7/10), alpha=0.1+memory − zero+memory = +1.8 ± 1.4 (7/10), not significant.

## Answer: no
- Pre-training does not make the 200-step search better. Used at full strength it is clearly worse than starting from zero, with the same sign at both reward scales and on 7 of 10 held-out assays.
- There is a small genuine transferable signal: pretrained weights rank held-out variants by fitness with Spearman 0.04-0.14 (positive on all 10 assays at 100 starts) versus about 0 when pre-trained on shuffled fitness. Navigation does not turn it into a gain: pretrained vs shuffled-fitness pre-training differs by only +2.1 ± 2.2 points.
- Shrinking the prior (alpha=0.1) removes the harm but not into a benefit (+0.5 ± 0.9 over a random walk; a post-hoc choice, so even this is optimistic).
- Quadrupling pre-training (400 starts per assay, 5 seeds) does not raise the zero-shot signal (Spearman still about 0.09) or fix navigation (pretrained - zero = -15.4 ± 2.8).
- Why it hurts (hypothesis, not tested): a miscalibrated value function plus a fixed low temperature makes the walker commit to wrong directions. Consistent with this, on KKA2, TPK1 and NUD15 true-fitness following (chemotaxis) gains +12 to +23 points at scale 0.45, yet learners that carry memory lose 7-28 points there, even without pre-training.
- Caveat: results use 8M embeddings (cheap on CPU). Larger embeddings were not tested for transfer. The web app was not changed.

Files: raw_scale0.45.csv, raw_scale1.0.csv, raw_scale0.45_pretrain400.csv (5 seeds), tables_*.md, raw_z1.0_first.csv (first run before the alpha arms and reward-scale option).
