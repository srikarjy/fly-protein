"""Social preview (1200x630) from results/benchmark/summary.json -> web/public/og.png."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
S = json.load(open(ROOT / "results/benchmark/summary.json"))
BG, INK, MUTED, LINE = "#fbfaf7", "#14181f", "#5a6270", "#e4e1d9"
SER = [("random_search", "Random search", "#7d838d", "--"), ("fly_memory", "Fly learner", "#d55e00", "-"), ("gpbo_ei", "GP-BO", "#0072b2", "-"), ("adalead_graph", "graph-adapted AdaLead", "#b8579a", "-"), ("zs_wt_650M", "ESM-2 zero-shot", "#00946b", "-")]
fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=BG)
fig.text(0.05, 0.86, "PROTEIN SEARCH UNDER A MEASUREMENT BUDGET", fontsize=10.5, color="#0b6b73", fontweight="bold")
fig.text(0.05, 0.60, "Better protein\nrepresentations\nwere not enough.", fontsize=33, color=INK, fontweight="bold", va="center", linespacing=1.12)
fig.text(0.05, 0.31, "Search strategy was the bottleneck.", fontsize=19, color="#d55e00", fontweight="bold")
fig.text(0.05, 0.215, "ESM-2 · ProteinGym · fly-inspired sparse coding\nGP-BO · evolutionary search · zero-shot PLMs", fontsize=12.5, color=MUTED, linespacing=1.5)
fig.text(0.05, 0.075, "10 assays × 20 seeds · 2,800 deterministic runs · negative results included", fontsize=11, color=MUTED)
ax = fig.add_axes([0.575, 0.2, 0.385, 0.6], facecolor=BG)
x = np.arange(1, 301)
ax.axvspan(50, 200, color="#0b6b73", alpha=0.07, lw=0)
for k, name, c, ls in SER:
    cu = S["curves"][k]
    ax.plot(x, np.array(cu["top1"])[:300], color=c, lw=2.6 if k == "fly_memory" else 2, ls=ls, label=name)
    ax.fill_between(x, np.array(cu["top1_lo"])[:300], np.array(cu["top1_hi"])[:300], color=c, alpha=0.0 if k.startswith("zs") else 0.12, lw=0)
ax.set_xlim(0, 300); ax.set_ylim(0, 100)
ax.set_xlabel("fitness measurements", color=MUTED, fontsize=10); ax.set_ylabel("runs that reached the assay's top 1%  (%)", color=MUTED, fontsize=10)
ax.text(125, 94, "50–200: informative range", color="#0b6b73", fontsize=9.5, ha="center", fontweight="bold")
leg = ax.legend(loc="lower right", fontsize=9.5, frameon=False, handlelength=2.2)
[t.set_color(c) for t, (_, _, c, _) in zip(leg.get_texts(), SER)]
ax.tick_params(colors=MUTED, labelsize=9); ax.grid(color=LINE, lw=0.8); [s.set_visible(False) for s in ax.spines.values()]
fig.text(0.575, 0.85, "Share of runs that found a top-1% variant", fontsize=11, color=INK, fontweight="bold")
fig.savefig(ROOT / "web/public/og.png", facecolor=BG)
print("wrote web/public/og.png")
