# Learner variants (FlyHash baseline cell: full d, 2000 KC, 5%, random wiring)

Tuned on seeds 100-104 (`tune.csv`), compared on held-out seeds 0-9 (`final.csv`), paired against the web app's learner. Metric: reach-top-1% gain over a random walk on the same graph and starts, in points.

Variants tried: learning rate x{0.3, 3, 10}, temperature {0.03, 0.3}, the "absolute" rule (learn each visited variant's fitness), and replay of 2/5/10 already-experienced transitions per step (never uses unvisited fitness).

Result: nothing beats the current learner reliably.
- Higher lr, lower temperature and the absolute rule are worse (the absolute rule is far below a random walk without memory).
- Best candidate, replay 5 + temperature 0.3: with memory +14.7 vs +12.8 pooled (paired +1.9 +- 2.1 SE, not significant). It helps 650M (+18.2 vs +8.8) but not 8M, so it is not shown to be a fix.
- Without memory every variant stays at random-walk level (gain about 0-3 points).
The web app was not changed.
