"""Export every number the website shows from the committed result files (nothing is typed in by hand).

    python scripts/export_web_results.py
Writes web/public/data/hero.json (first-load: headline cards + main figure) and web/public/data/results.json (lazy-loaded sections).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = ROOT / "web/public/data"
OUT.mkdir(parents=True, exist_ok=True)
S = json.load(open(R / "benchmark/summary.json"))
B = [str(b) for b in S["budgets"]]
ci = lambda t: [round(float(x), 1) for x in t]

# ---------------- hero: headline cards + main figure
HERO = ["random_search", "fly_memory", "zs_wt_650M", "gpbo_ei", "adalead_graph"]
top1 = S["agg"]["top1"]
pair = S["paired"]["fly"]
abl = pd.read_csv(R / "ablation/raw.csv")
fh650 = abl[(abl.model == "650M") & (abl.method == "flyhash") & (abl.d == 1280) & (abl.n_kc == 2000) & (abl.sparsity == 0.05)].fit_rho
cc650 = abl[(abl.model == "650M") & (abl.method == "centered_cosine") & (abl.d == 1280)].fit_rho
hero = dict(
    cards=[
        dict(id="adalead", value=f"+{pair['adalead_graph']['50'][0]:.1f}", unit="points", ci=ci(pair["adalead_graph"]["50"][1:]),
             label="graph-adapted AdaLead over the fly learner: P(reach assay top 1%) at 50 queries",
             sub=f"GP-BO: +{pair['gpbo_ei']['50'][0]:.1f} {ci(pair['gpbo_ei']['50'][1:])}"),
        dict(id="zeroshot", value=f"{top1['zs_wt_650M']['50'][0]:.0f}%", unit="", ci=ci(top1["zs_wt_650M"]["50"][1:]),
             label="ESM-2 650M zero-shot ranking reaches the assay top 1% within its first 50 measured candidates",
             sub=f"fly learner {top1['fly']['50'][0]:.1f}% {ci(top1['fly']['50'][1:])}; random search {top1['random_search']['50'][0]:.1f}%. The ranking uses no observed fitness."),
        dict(id="scale", value=f"{S['n_assays']} × {S['n_seeds']}", unit="", ci=None, label="unrelated ProteinGym assays × seeds; 14 methods; 2,800 logged runs", sub="budgets of 50, 100, 200 and 500 measured variants"),
        dict(id="flyhash", value=f"{fh650.mean():.3f}", unit="vs " + f"{cc650.mean():.3f}", ci=None,
             label="FlyHash vs dense centered cosine: how well kNN neighbours predict fitness (Spearman, ESM-2 650M)",
             sub=f"FlyHash ±{fh650.std(ddof=1):.3f} over 10 wirings. The sparse code keeps {100 * fh650.mean() / cc650.mean():.0f}% of the dense signal."),
    ],
    figure=dict(budgets=S["budgets"], names={m: S["names"][m] for m in HERO}, curves={m: {k: v[:500] for k, v in S["curves"][m].items()} for m in HERO}),
    table=dict(top1={m: S["agg"]["top1"][m] for m in HERO}, pct={m: S["agg"]["pct"][m] for m in HERO}),
    meta=dict(n_assays=S["n_assays"], n_seeds=S["n_seeds"], assays=S["assays"]),
)
(OUT / "hero.json").write_text(json.dumps(hero, separators=(",", ":")))

# ---------------- benchmark section
ALL = list(S["names"])
bench = dict(names=S["names"], budgets=S["budgets"], top1=S["agg"]["top1"], top01=S["agg"]["top0.1"], pct=S["agg"]["pct"], auc=S["agg"]["auc"],
             paired_fly=S["paired"]["fly"], paired_random=S["paired"]["random_search"],
             per_assay={m: {a: {k: S["per_assay"][m][a][k] for k in ("top1@50", "top1@100", "top1@200", "top1@500", "pct@50", "pct@100", "pct@200", "pct@500")} for a in S["assays"]} for m in ALL},
             assays=S["assays"], curves={m: {k: v for k, v in S["curves"][m].items()} for m in ALL})

# ---------------- representation + ablation
rep = {}
full = {"8M": 320, "150M": 640, "650M": 1280}
ms = lambda s: [round(float(s.mean()), 4), round(float(s.std(ddof=1)) if len(s) > 1 else 0.0, 4)]
rep["models"] = {m: {meth: dict(fit_rho=ms(abl[(abl.model == m) & (abl.method == meth) & (abl.d == full[m]) & (abl.n_kc.isin([0, 2000])) & (abl.sparsity.isna() | (abl.sparsity == 0.05))].fit_rho),
                              recall15=ms(abl[(abl.model == m) & (abl.method == meth) & (abl.d == full[m]) & (abl.n_kc.isin([0, 2000])) & (abl.sparsity.isna() | (abl.sparsity == 0.05))].recall15))
                     for meth in ("dense_cosine", "centered_cosine", "simhash", "flyhash")} for m in full}
fl = abl[(abl.method == "flyhash") & (abl.sparsity == 0.05)]
rep["dim_sweep"] = {m: [[int(d), ms(fl[(fl.model == m) & (fl.d == d) & (fl.n_kc == 2000)].fit_rho)[0]] for d in sorted(fl[fl.model == m].d.unique())] for m in full}
rep["kc_sweep"] = {m: [[int(k), ms(fl[(fl.model == m) & (fl.d == full[m]) & (fl.n_kc == k)].fit_rho)[0], ms(fl[(fl.model == m) & (fl.d == full[m]) & (fl.n_kc == k)].recall15)[0]] for k in (2000, 4000, 8000, 16000)] for m in full}
fs = abl[(abl.method == "flyhash")]
rep["sparsity_sweep"] = {m: [[sp, ms(fs[(fs.model == m) & (fs.d == full[m]) & (fs.n_kc == 2000) & (fs.sparsity == sp)].fit_rho)[0]] for sp in (0.02, 0.05, 0.1)] for m in full}
rw = abl[abl.method == "random_walk"].set_index(["model", "seed"]).nav_reach_top1pct * 100
cell = fl[fl.navcarry_reach_top1pct.notna()].copy()
cell["gain"] = [r.navcarry_reach_top1pct * 100 - rw[(r.model, r.seed)] for r in cell.itertuples()]
g = cell.groupby(["model", "d", "n_kc"]).agg(gain=("gain", "mean"), fit=("fit_rho", "mean")).reset_index()
rep["nav_vs_representation"] = [dict(model=r.model, ratio=round(r.n_kc / r.d, 3), gain=round(r.gain, 2), fit_rho=round(r.fit, 4), d=int(r.d), n_kc=int(r.n_kc)) for r in g.itertuples()]
rep["note"] = "FlyHash 5% sparsity, random wiring; gain = reach-top-1% points over a random walk with memory kept (50 starts x 200 steps); 10 seeds."

# ---------------- learner ablation
tune = pd.read_csv(R / "learner/tune.csv")
final = pd.read_csv(R / "learner/final.csv")
learner = dict(tuning=[dict(config=c, nomem=round(float(tune[(tune.config == c) & (~tune.carry)].gain.mean()), 1), mem=round(float(tune[(tune.config == c) & (tune.carry)].gain.mean()), 1))
                       for c in tune.config.unique()])
rows = []
for carry in (False, True):
    base = final[(final.config == "baseline") & (final.carry == carry)].set_index(["model", "seed"]).gain
    for c in final.config.unique():
        a = final[(final.config == c) & (final.carry == carry)].set_index(["model", "seed"]).gain
        d = a - base
        rows.append(dict(config=c, memory=bool(carry), gain=round(float(a.mean()), 1), diff=round(float(d.mean()), 1), se=round(float(d.std(ddof=1) / np.sqrt(len(d))), 1)))
learner["held_out"] = rows

# ---------------- transfer
def transfer(name):
    d = pd.read_csv(R / f"transfer/{name}.csv")
    rwk = d[d.arm == "random"].set_index(["heldout", "seed"]).top1
    d["gain"] = [r.top1 - rwk[(r.heldout, r.seed)] for r in d.itertuples()]
    P = d.pivot_table(index=["heldout", "seed"], columns="arm", values="gain")
    arms = {a: [round(float(P[a].mean()), 1), round(float(P[a].std(ddof=1) / np.sqrt(len(P))), 1)] for a in P.columns}
    pr = lambda a, b: [round(float((P[a] - P[b]).mean()), 1), round(float((P[a] - P[b]).std(ddof=1) / np.sqrt(len(P))), 1)]
    zs = d[d.arm.isin(["pre", "shuf_pre"])].pivot_table(index="heldout", columns="arm", values="zeroshot_rho")
    return dict(n_pairs=len(P), arms=arms, pre_minus_zero=pr("pre", "zero"), pre_minus_shuffled=pr("pre", "shuf_pre"), pre_mem_minus_zero_mem=pr("pre_mem", "zero_mem"),
                zeroshot={h.split("_")[0]: [round(float(zs.loc[h, "pre"]), 3), round(float(zs.loc[h, "shuf_pre"]), 3)] for h in zs.index})
tr = dict(main=transfer("raw_scale0.45"), reward_scale_1=transfer("raw_scale1.0"), pretrain_x4=transfer("raw_scale0.45_pretrain400"))

# ---------------- hemibrain
hb = pd.read_csv(R / "hemibrain/raw.csv")
hemi = {}
for m in ("8M", "150M", "650M"):
    h = hb[hb.model == m]
    row = {}
    for arm in ("random_2000", "random_1785", "hemibrain"):
        s = h[h.arm == arm]
        row[arm] = dict(fit_rho=ms(s.fit_rho)[0], recall15=ms(s.recall15)[0], gain=ms(s.gain)[0], gain_carry=ms(s.gain_carry)[0])
    a = h[h.arm == "hemibrain"].set_index("seed")
    b = h[h.arm == "random_1785"].set_index("seed")
    row["diff"] = {c: [round(float((a[c] - b[c]).mean()), 3), round(float((a[c] - b[c]).std(ddof=1) / np.sqrt(len(a))), 3)] for c in ("fit_rho", "recall15", "gain", "gain_carry")}
    hemi[m] = row

(OUT / "results.json").write_text(json.dumps(dict(benchmark=bench, representation=rep, learner=learner, transfer=tr, hemibrain=hemi), separators=(",", ":")))
for p in sorted(OUT.glob("*.json")):
    print(p.name, f"{p.stat().st_size / 1e3:.0f} KB")
print(json.dumps(hero["cards"], indent=1)[:1800])
