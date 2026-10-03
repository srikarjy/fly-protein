# Real hemibrain PN->KC wiring vs random wiring

`python scripts/hemibrain_ablation.py` (TEM-1, ESM-2 8M / 150M / 650M embeddings, 10 seeds, 5% sparsity, learner rule unchanged). This is a reproducible Python re-implementation of the earlier browser-side evaluation; exact numbers differ slightly from that evaluation because random streams differ.

Wiring: hemibrain:v1.2.1 PN->KC synapse counts onto the calyx (135 PNs, 1,785 KCs, 12,012 edges, neuPrint). The embedding is reduced to 135 channels with a fixed seeded Gaussian projection. Control = random wiring with the same 1,785 KC code length.

Result (hemibrain - random_1785, mean +- SE over 10 seeds):
| embedding | kNN fitness Spearman | navigation gain over random walk, no memory (points) | with memory (points) |
|---|---|---|---|
| 8M | -0.032 +- 0.005 | -0.6 +- 2.0 | -5.0 +- 3.4 |
| 150M | -0.037 +- 0.002 | -1.2 +- 3.5 | -1.2 +- 2.8 |
| 650M | -0.040 +- 0.003 | -3.2 +- 3.3 | -8.4 +- 5.2 |

Neighbourhood preservation (recall@15 of the walker graph) is also lower: 0.54-0.56 vs 0.70-0.72. The real circuit gives no measurable advantage; representation quality is slightly worse, navigation is level or nominally worse. Caveat: the input stage (random projection to 135 channels) is our choice, not a model of the fly's antennal lobe.
