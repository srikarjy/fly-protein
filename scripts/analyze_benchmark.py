"""Metrics, tables and headline plots from results/benchmark/queries.npz (all computed from the query logs).

Per-assay percentile normalisation only: raw DMS scores are never averaged across assays.
Unit of replication = assay (seed means within assay); intervals are 95% bootstrap CIs over assays.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from embed_assays import ASSAYS  # noqa: E402

from flyprotein.bench import curve_percentiles, load_assay_data  # noqa: E402

OUT = ROOT / "results/benchmark"
BUDGETS = [50, 100, 200, 500]
Z = np.load(OUT / "queries.npz")
runs = {}
for k in Z.files:
    a, m, s = k.split("|")
    runs[(a, m, int(s))] = Z[k]
methods = sorted({m for _, m, _ in runs})
seeds = sorted({s for _, _, s in runs})
FIT = {a: load_assay_data(a) for a in ASSAYS}

NAMES = {
    "random_search": "Random search", "random_walk": "Random walk (graph)", "greedy_local": "Greedy local search",
    "regevo": "Regularized evolution", "gpbo_ei": "GP-BO (Matern-5/2, EI)", "adalead_graph": "graph-adapted AdaLead",
    "fly": "Fly learner", "fly_memory": "Fly learner + memory", "fly_best_tested": "Fly best tested (replay5, T=0.3, memory)",
    "zs_wt_8M": "ESM-2 8M zero-shot (WT-marginal)", "zs_wt_150M": "ESM-2 150M zero-shot (WT-marginal)", "zs_wt_650M": "ESM-2 650M zero-shot (WT-marginal)",
    "zs_mm_8M": "ESM-2 8M zero-shot (masked-marginal)", "zs_mm_150M": "ESM-2 150M zero-shot (masked-marginal)",
}
methods = [m for m in NAMES if m in methods]

# ---- per-run curves (n_runs, 500) and scalar metrics
curves = {}
for (a, m, s), log in runs.items():
    curves[(a, m, s)] = curve_percentiles(log, FIT[a], 500)


def metrics(c):
    r = {}
    for b in BUDGETS:
        r[f"pct@{b}"] = c[b - 1]
        r[f"top1@{b}"] = float(c[b - 1] >= 99.0) * 100
        r[f"top0.1@{b}"] = float(c[b - 1] >= 99.9) * 100
        r[f"auc@{b}"] = np.nanmean(c[:b])
    return r


rows = [dict(assay=a, method=m, seed=s, **metrics(c)) for (a, m, s), c in curves.items()]
raw = pd.DataFrame(rows)
raw.to_csv(OUT / "per_run_metrics.csv", index=False)
per_assay = raw.groupby(["assay", "method"]).mean(numeric_only=True).drop(columns="seed").reset_index()
per_assay.to_csv(OUT / "per_assay_metrics.csv", index=False)

rng = np.random.default_rng(0)


def boot(vals, fn=np.mean, n=4000):
    v = np.asarray(vals, float)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    b = fn(v[idx], axis=1)
    return fn(v), np.percentile(b, 2.5), np.percentile(b, 97.5)


def fmt(t, d=1):
    return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"


md = []
r4 = lambda t: [round(float(x), 4) for x in t]
SUMM = dict(agg={}, paired={})
# ---- aggregate tables
for metric, title, d in [("pct@", "Best fitness percentile found (mean over assays; [95% bootstrap CI over assays])", 2),
                         ("top1@", "P(reach assay top 1%) in % (mean over assays; [95% CI])", 1),
                         ("top0.1@", "P(reach assay top 0.1%) in % (mean over assays; [95% CI])", 1),
                         ("auc@", "Area under best-percentile-vs-query curve, normalised (mean over assays; [95% CI])", 2)]:
    md.append(f"## {title}\n\n| method | " + " | ".join(f"B={b}" for b in BUDGETS) + " |\n|---|" + "---|" * len(BUDGETS))
    for m in methods:
        cells = []
        for b in BUDGETS:
            v = per_assay[per_assay.method == m][f"{metric}{b}"]
            t = boot(v)
            SUMM["agg"].setdefault(metric[:-1], {}).setdefault(m, {})[b] = r4(t)
            cells.append(fmt(t, d))
        md.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    md.append("")

# ---- paired differences vs references (assay-level paired means)
def paired(a_m, b_m, metric):
    A = per_assay[per_assay.method == a_m].set_index("assay")[metric]
    B = per_assay[per_assay.method == b_m].set_index("assay")[metric]
    return boot((A - B).loc[ASSAYS].values)


for ref in ["random_search", "fly", "greedy_local"]:
    md.append(f"## Paired difference in P(top 1%) (points) vs **{NAMES[ref]}** (mean over assays [95% CI]); positive = method better\n\n| method | " + " | ".join(f"B={b}" for b in BUDGETS) + " |\n|---|" + "---|" * len(BUDGETS))
    for m in methods:
        if m == ref:
            continue
        ts = {b: paired(m, ref, f"top1@{b}") for b in BUDGETS}
        SUMM["paired"].setdefault(ref, {})[m] = {b: r4(t) for b, t in ts.items()}
        md.append(f"| {NAMES[m]} | " + " | ".join(fmt(ts[b]) for b in BUDGETS) + " |")
    md.append("")

# ---- per assay tables at B=100 and 500
for b in (100, 500):
    md.append(f"## Per assay: P(top 1%) in % at B={b} (mean over {len(seeds)} seeds)\n")
    P = per_assay.pivot(index="method", columns="assay", values=f"top1@{b}").loc[methods, ASSAYS].round(0)
    P.index = [NAMES[m] for m in P.index]
    P.columns = [c.split("_")[0] for c in P.columns]
    md.append(P.to_string() + "\n")
    md.append(f"## Per assay: best percentile at B={b} (mean over seeds)\n")
    P = per_assay.pivot(index="method", columns="assay", values=f"pct@{b}").loc[methods, ASSAYS].round(2)
    P.index = [NAMES[m] for m in P.index]
    P.columns = [c.split("_")[0] for c in P.columns]
    md.append(P.to_string() + "\n")

# ---- zero-shot sanity: Spearman of the score vs measured fitness (evaluation only)
(OUT / "tables.md").write_text("\n".join(md))

# ---- plots
KEY = ["random_search", "random_walk", "greedy_local", "regevo", "gpbo_ei", "adalead_graph", "fly_memory", "zs_wt_650M", "zs_mm_150M"]
KEY = [m for m in KEY if m in methods]
COL = dict(zip(KEY, ["#868e96", "#adb5bd", "#f08c00", "#7048e8", "#e03131", "#c2255c", "#1971c2", "#2f9e44", "#0b7285"]))
bs = np.arange(1, 501)
mean_curve = {}
for m in KEY:
    per = np.stack([np.nanmean([curves[(a, m, s)] for s in seeds], axis=0) for a in ASSAYS])  # (assay, 500)
    mean_curve[m] = per
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for m in KEY:
    per = mean_curve[m]
    mu = per.mean(0)
    lo, hi = np.percentile(per[rng.integers(0, len(per), size=(1000, len(per)))].mean(1), [2.5, 97.5], axis=0)
    ax[0].plot(bs, mu, color=COL[m], label=NAMES[m]); ax[0].fill_between(bs, lo, hi, color=COL[m], alpha=0.12)
    ax[1].plot(bs, 100 - mu, color=COL[m])
    t1 = np.stack([np.mean([curves[(a, m, s)] >= 99 for s in seeds], axis=0) for a in ASSAYS]) * 100
    ax[2].plot(bs, t1.mean(0), color=COL[m])
ax[0].set_ylim(94, 100.05); ax[0].set_title("Best fitness percentile found vs queries"); ax[0].set_ylabel("percentile of best queried variant")
ax[1].set_yscale("log"); ax[1].set_title("Remaining gap to the best variant, 100 - percentile (log; lower is better)")
ax[2].set_title("P(reached assay top 1%)"); ax[2].set_ylabel("%")
for a_ in ax:
    a_.set_xlabel("fitness queries"); a_.set_xscale("log"); a_.axvline(50, color="k", lw=0.3); a_.axvline(200, color="k", lw=0.3)
ax[0].legend(fontsize=7, loc="lower right")
fig.suptitle(f"Query-budgeted search on {len(ASSAYS)} ProteinGym assays (mean over assays, {len(seeds)} seeds each; band = 95% bootstrap CI over assays)")
fig.tight_layout(); fig.savefig(OUT / "headline_curves.png", dpi=130)

fig, axs = plt.subplots(2, 5, figsize=(20, 7), sharex=True, sharey=True)
for ax_, a in zip(axs.ravel(), ASSAYS):
    for m in KEY:
        ax_.plot(bs, np.nanmean([curves[(a, m, s)] for s in seeds], axis=0), color=COL[m], lw=1.1)
    ax_.set_title(a.split("_")[0] + " " + a.split("_")[-1], fontsize=9); ax_.set_xscale("log"); ax_.set_ylim(90, 100.05)
fig.supxlabel("fitness queries"); fig.supylabel("best percentile found")
fig.tight_layout(); fig.savefig(OUT / "per_assay_curves.png", dpi=120)

fig, ax = plt.subplots(1, 4, figsize=(20, 4.6), sharey=True)
for a_, b in zip(ax, BUDGETS):
    vals = [(NAMES[m], boot(per_assay[per_assay.method == m][f"top1@{b}"])) for m in methods]
    y = np.arange(len(vals))
    a_.barh(y, [v[1][0] for v in vals], xerr=[[v[1][0] - v[1][1] for v in vals], [v[1][2] - v[1][0] for v in vals]], color="#4c6ef5", alpha=0.8)
    a_.set_yticks(y); a_.set_yticklabels([v[0] for v in vals], fontsize=8); a_.invert_yaxis(); a_.set_title(f"P(top 1%) at B={b}"); a_.set_xlabel("%")
fig.tight_layout(); fig.savefig(OUT / "top1_by_budget.png", dpi=120)

# ---- machine-readable summary for the website (same numbers as tables.md)
import json
summ = dict(budgets=BUDGETS, n_assays=len(ASSAYS), n_seeds=len(seeds), assays=ASSAYS, names={m: NAMES[m] for m in methods}, agg=SUMM["agg"], paired=SUMM["paired"], per_assay={}, curves={})
summ["per_assay"] = {m: {a: {k: round(float(v), 3) for k, v in per_assay[(per_assay.assay == a) & (per_assay.method == m)].iloc[0].items() if k not in ("assay", "method")} for a in ASSAYS} for m in methods}
for m in methods:
    per = np.stack([np.nanmean([curves[(a, m, s)] for s in seeds], axis=0) for a in ASSAYS])
    t1 = np.stack([np.mean([curves[(a, m, s)] >= 99 for s in seeds], axis=0) for a in ASSAYS]) * 100
    ix = rng.integers(0, len(per), size=(1000, len(per)))
    summ["curves"][m] = {"pct": per.mean(0).round(3).tolist(), "pct_lo": np.percentile(per[ix].mean(1), 2.5, axis=0).round(3).tolist(), "pct_hi": np.percentile(per[ix].mean(1), 97.5, axis=0).round(3).tolist(),
                         "top1": t1.mean(0).round(2).tolist(), "top1_lo": np.percentile(t1[ix].mean(1), 2.5, axis=0).round(2).tolist(), "top1_hi": np.percentile(t1[ix].mean(1), 97.5, axis=0).round(2).tolist()}
(OUT / "summary.json").write_text(json.dumps(summ, separators=(",", ":")))
print("\n".join(md[:60]))
