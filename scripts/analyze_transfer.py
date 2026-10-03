"""Tables from results/transfer/raw.csv: gains over random walk (points, reach top 1%), paired differences."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "results/transfer"
name = sys.argv[1] if len(sys.argv) > 1 else "raw"
d = pd.read_csv(OUT / f"{name}.csv")
rw = d[d.arm == "random"].set_index(["heldout", "seed"]).top1
d["gain"] = [r.top1 - rw[(r.heldout, r.seed)] for r in d.itertuples()]
P = d.pivot_table(index=["heldout", "seed"], columns="arm", values="gain")
short = lambda a: a.split("_")[0]
ARMS = ["chemotaxis", "zero", "zero_mem", "pre", "pre_mem", "pre_frozen", "shuf_pre", "shuf_pre_mem", "pre_a0.1", "pre_a0.3", "pre_mem_a0.1", "pre_mem_a0.3"]
out = []

def mse(x):
    return f"{x.mean():+.1f} ± {x.std(ddof=1) / np.sqrt(len(x)):.1f}"

NS = d.seed.nunique()
out.append(f"## 1. Gain over random walk, reach top 1% (points). Pooled over {d.heldout.nunique()} held-out assays x {NS} seeds; ± = SE over the {d.heldout.nunique() * NS} (assay, seed) pairs\n")
out.append("| arm | gain over random | gain, top-10% reach |\n|---|---|---|")
P10 = d.assign(g10=d.top10 - d.set_index(["heldout", "seed"]).index.map(d[d.arm == "random"].set_index(["heldout", "seed"]).top10)).pivot_table(index=["heldout", "seed"], columns="arm", values="g10")
for a in ARMS:
    out.append(f"| {a} | {mse(P[a])} | {mse(P10[a])} |")

out.append("\n## 2. Paired differences (points, top 1%); ± = SE over (assay, seed) pairs; 'wins' = held-out assays where the seed-mean difference is > 0\n")
out.append("| comparison | mean diff | wins (of held-out assays) |\n|---|---|---|")
for a, b in [("pre", "zero"), ("pre_mem", "zero_mem"), ("pre_mem", "zero"), ("pre_frozen", "zero"), ("pre", "shuf_pre"), ("pre_mem", "shuf_pre_mem"), ("zero_mem", "zero"), ("pre_a0.1", "zero"), ("pre_a0.3", "zero"), ("pre_mem_a0.1", "zero_mem"), ("pre_mem_a0.3", "zero_mem")]:
    diff = P[a] - P[b]
    wins = (diff.groupby("heldout").mean() > 0).sum()
    out.append(f"| {a} − {b} | {mse(diff)} | {wins} |")

out.append("\n## 3. Per held-out assay: gain over random (mean over seeds)\n")
G = P.groupby("heldout").mean()[["chemotaxis", "zero", "zero_mem", "pre", "pre_mem", "pre_a0.1", "pre_mem_a0.1", "shuf_pre"]].round(1)
G.index = [short(i) + " " + i.split("_")[-1] for i in G.index]
out.append(G.to_string())

out.append("\n## 4. Zero-shot value of pretrained weights on the held-out assay (Spearman of w·code vs fitness, mean ± std over seeds)\n")
Z = d[d.arm.isin(["pre", "shuf_pre"])].pivot_table(index="heldout", columns="arm", values="zeroshot_rho", aggfunc=["mean", "std"]).round(3)
Z.index = [short(i) + " " + i.split("_")[-1] for i in Z.index]
out.append(Z.to_string())
(OUT / f"tables_{name}.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
