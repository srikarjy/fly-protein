# Representation / compression ablation (learner unchanged)

Question: does ESM-2 650M underperform because its 1280-d embedding is compressed too aggressively by the fly-inspired projection?

Setup: cached TEM-1 (BLAT_ECOLX_Firnberg_2014) embeddings from ESM-2 8M / 150M / 650M, no regeneration. Random wiring. Input (PN) dim d in {64,128,256,512,1024,full}, KC count in {2000,4000,8000,16000}, sparsity {2,5,10}%, 10 seeds (wiring, projection and start points all vary). Methods: dense cosine, centered cosine, SimHash (same bits as KC count), FlyHash, FlyHash + memory carried across starts. Navigation uses the web app's walker (50 starts x 200 steps, T=0.1, lr=0.1), ported in `src/flyprotein/navigate.py`; its reward-prediction-error rule is applied to every representation as `w += lr*delta*(phi_next-phi_cur)/(|phi_next|^2+|phi_cur|^2)`, which is exactly the web rule for binary codes (tested). Full tables: `tables.md`; raw: `raw.csv`; plots: `*.png`.

## Answer: no
1. **The premise does not hold for random wiring.** The pipeline never projects the embedding down: FlyHash reads all d dimensions. What shrinks with model size is the expansion ratio KC/d (6.25 / 3.1 / 1.56 at 2000 KC).
2. **650M has the best representation, and FlyHash keeps it.** Leave-one-out kNN fitness Spearman: 8M 0.66, 150M 0.79, 650M 0.82 (dense centered cosine 0.69 / 0.81 / 0.83). Cutting d from 1280 to 64 costs about 0.04; raising KC from 2000 to 16000 changes it by about +0.01.
3. **Matching the expansion ratio does not recover navigation.** 650M at 8000 KC (ratio 6.25, same as 8M at 2000): representation is far better (Spearman +0.16, recall@15 +0.11), but navigation gain over a random walk is lower or equal (7.2 vs 14.2 points, paired difference -7.0 +- 11.5). Across the d x KC grid, representation quality and expansion ratio do not predict navigation gain (rank correlations -0.3 to +0.15 within a model).
4. **Representation is not what limits navigation.** Without memory, every representation, including dense centered cosine, is statistically indistinguishable from a random walk in 200 steps. With memory, centered cosine, SimHash and FlyHash are within noise of each other on each model. FlyHash is not worse than the dense baselines, so the fly-style projection is not the loss.
5. **650M's weaker navigation shows up in every representation** (centered cosine 8.2 +- 24, SimHash 9.2 +- 15, FlyHash 8.8 +- 15 vs 13-20 at 8M/150M), so the cause is in the walker/graph/readout interaction, not in FlyHash compression. Caveat: the spread is large (SE about 4-7 points on 10 seeds), so the 650M deficit is suggestive, not established.

## Other findings
- Uncentered dense cosine gives zero gain even with memory; centering matters for the readout (FlyHash centers internally via `fit`).
- recall@15 against the walker's graph rises with KC count (0.71 to 0.85 for FlyHash at full d) but this does not translate into navigation.
- The three models use different kNN graphs, so their random-walk baselines differ (51 / 65 / 53% reach top 1%). Compare gains over the random walk, not raw reach rates.
- More sparsity (10%) is slightly better for recall; no reliable navigation effect.
