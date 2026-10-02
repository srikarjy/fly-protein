# Fly Walker (web)

Static page: a map of ~4.8k TEM-1 beta-lactamase variants (ProteinGym `BLAT_ECOLX_Firnberg_2014`, measured fitness) and a fly that walks the real 15-NN graph in ESM-2 embedding space.

```bash
npm install
npm run dev      # http://localhost:5173
npm test         # vitest: softmax step, reward-gated update, fly logic
npm run eval     # 50 starts x 200 steps, 5 seeds, on the real landscape
npm run build    # static site in dist/
```

Data comes from `../prep/build_data.py` (writes `public/data.json`).

## Modes
- **Chemotaxis**: baseline, no brain model. Sees true neighbor fitness.
- **Mushroom body**: FlyHash Kenyon codes (random wiring), plastic KC->output weights, dopamine-like error. Never sees neighbor fitness.
- **Random walk**: control.
- **Hemibrain wiring**: not built (M5).

## Eval results (npm run eval, mean of 5 seeds)
| Mode | Median best | Top 10% reached | Top 1% reached |
|---|---|---|---|
| Random walk | 1.428 | 86% | 53% |
| Chemotaxis | 1.679 | 95% | 80% |
| Mushroom body (fresh each start) | 1.436 | 82% | 50% |
| Mushroom body (weights kept across starts) | 1.625 | 88% | 66% |

Within one 200-step run the learner is no better than a random walk. With synapses kept across starts it learns, but chemotaxis (which gets neighbor fitness for free) still wins. "Top 10%" is a weak test: 200 steps covers ~4% of the space.
