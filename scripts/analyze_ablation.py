"""Tables and plots from results/ablation/raw.csv (mean +- std over seeds)."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "results/ablation"
df = pd.read_csv(OUT / "raw.csv")
MODELS = ["8M", "150M", "650M"]
BASE_KC, BASE_SP = 2000, 0.05
for c in ["nav_reach_top1pct", "nav_reach_top10pct", "navcarry_reach_top1pct", "navcarry_reach_top10pct"]:
    df[c] = df[c] * 100


# paired navigation gain over a random walk on the SAME graph and starts (graphs differ by embedding)
rw = df[df.method == "random_walk"].set_index(["model", "seed"])["nav_reach_top1pct"]
df["rw1"] = [rw[(m, sd)] for m, sd in zip(df.model, df.seed)]
df["gain_nav"] = df.nav_reach_top1pct - df.rw1
df["gain_carry"] = df.navcarry_reach_top1pct - df.rw1


def ms(s, f="{:.1f}"):
    s = s.dropna()
    return "n/a" if s.empty else (f + " ± " + f).format(s.mean(), s.std(ddof=1))


def agg(frame, keys, cols, fmt=None):
    g = frame.groupby(keys, dropna=False, sort=False)
    out = g[cols].agg(lambda s: ms(s, "{:.3f}" if s.name in ("recall15", "fit_rho") else "{:.1f}" if ("reach" in s.name or "gain" in s.name) else "{:.3f}"))
    return out.reset_index()


def md(frame):
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in frame.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(lines)


METRICS = ["recall15", "fit_rho", "nav_reach_top1pct", "navcarry_reach_top1pct", "gain_nav", "gain_carry"]
sections = []

# T1: baseline per model
t1 = []
for m in MODELS:
    s = df[df.model == m]
    D = int(s.D.iloc[0])
    full = s[(s.d == D) | s.method.isin(["random_walk", "chemotaxis"])]
    full = full[(full.n_kc.isin([0, BASE_KC])) & (full.sparsity.isna() | (full.sparsity == BASE_SP))]
    order = ["random_walk", "chemotaxis", "dense_cosine", "centered_cosine", "simhash", "flyhash"]
    a = agg(full, ["method"], METRICS)
    a["method"] = pd.Categorical(a["method"], order)
    a = a.sort_values("method")
    a.insert(0, "model", f"{m} (D={D})")
    t1.append(a)
sections.append(("1. Baseline (full dimension, 2000 KC, 5% sparsity)", md(pd.concat(t1))))

# T2: PN/input dimension sweep at 2000 KC, 5%
t2 = []
for m in MODELS:
    s = df[(df.model == m) & (df.method.isin(["dense_cosine", "centered_cosine", "simhash", "flyhash"]))]
    s = s[(s.n_kc.isin([0, BASE_KC])) & (s.sparsity.isna() | (s.sparsity == BASE_SP))]
    a = agg(s, ["method", "d"], METRICS)
    a.insert(0, "model", m)
    t2.append(a)
sections.append(("2. Input (PN) dimension sweep at 2000 KC / 5%", md(pd.concat(t2))))

# T3: KC sweep at full d, 5%
t3 = []
for m in MODELS:
    s = df[(df.model == m)]
    D = int(s.D.iloc[0])
    s = s[(s.d == D) & (s.method.isin(["simhash", "flyhash"])) & (s.sparsity.isna() | (s.sparsity == BASE_SP))]
    a = agg(s, ["method", "n_kc"], METRICS)
    a.insert(1, "ratio KC/d", a["n_kc"].map(lambda k: f"{k / D:.2f}"))
    a.insert(0, "model", m)
    t3.append(a)
sections.append(("3. KC-count sweep at full dimension, 5% sparsity", md(pd.concat(t3))))

# T4: sparsity at full d
t4 = []
for m in MODELS:
    s = df[(df.model == m) & (df.method == "flyhash")]
    D = int(s.D.iloc[0])
    s = s[s.d == D]
    a = agg(s, ["n_kc", "sparsity"], METRICS)
    a.insert(0, "model", m)
    t4.append(a)
sections.append(("4. Sparsity sweep (FlyHash, full dimension; navigation only measured at 2000 KC and at 5%)", md(pd.concat(t4))))

# T5: does matching the expansion ratio rescue 650M? (paired by seed)
fh = df[(df.method == "flyhash") & (df.sparsity == BASE_SP)]
def cell(m, d, k, col):
    s = fh[(fh.model == m) & (fh.d == d) & (fh.n_kc == k)].set_index("seed")[col]
    return s
lines = ["| comparison | metric | A | B | A − B (paired, mean ± std) |", "|---|---|---|---|---|"]
for (ma, da, ka), (mb, db, kb) in [(("650M", 1280, 8000), ("8M", 320, 2000)), (("650M", 1280, 2000), ("8M", 320, 2000)), (("150M", 640, 4000), ("8M", 320, 2000)), (("650M", 1280, 16000), ("8M", 320, 2000))]:
    for col in ["fit_rho", "recall15", "gain_nav", "gain_carry"]:
        A, B = cell(ma, da, ka, col), cell(mb, db, kb, col)
        d_ = (A - B).dropna()
        f = "{:.3f}" if col in ("fit_rho", "recall15") else "{:.1f}"
        lines.append(f"| {ma} d={da} KC={ka} (ratio {ka / da:.2f}) vs {mb} d={db} KC={kb} (ratio {kb / db:.2f}) | {col} | {f.format(A.mean())} | {f.format(B.mean())} | {(f + ' ± ' + f).format(d_.mean(), d_.std(ddof=1))} |")
sections.append(("5. Does matching the KC/d expansion ratio close the gap to 8M?", "\n".join(lines)))

# T6: does any representation metric explain navigation gain? (Spearman across cells of the grid)
from scipy.stats import spearmanr
lines = ["| model | n cells | rho(fit_rho, gain_carry) | rho(recall15, gain_carry) | rho(log KC/d, gain_carry) | rho(log KC/d, fit_rho) |", "|---|---|---|---|---|---|"]
for m in MODELS + ["all"]:
    c = fh[(fh.model == m) if m != "all" else slice(None)].dropna(subset=["gain_carry"]).groupby(["model", "d", "n_kc"]).mean(numeric_only=True).reset_index()
    c["lr"] = np.log(c.ratio)
    r = lambda a, b: f"{spearmanr(c[a], c[b]).statistic:+.2f}"
    lines.append(f"| {m} | {len(c)} | {r('fit_rho','gain_carry')} | {r('recall15','gain_carry')} | {r('lr','gain_carry')} | {r('lr','fit_rho')} |")
sections.append(("6. Rank correlations across (d, n_KC) cells, FlyHash 5% (cell = mean over seeds)", "\n".join(lines)))

(OUT / "tables.md").write_text("\n\n".join(f"## {t}\n\n{b}" for t, b in sections) + "\n")

# ---- plots
col = {"8M": "#1c7ed6", "150M": "#2f9e44", "650M": "#e8590c"}
fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
for m in MODELS:
    s = fh[fh.model == m].groupby(["d", "n_kc"]).mean(numeric_only=True).reset_index()
    ax[0].scatter(s.ratio, s.fit_rho, c=col[m], label=m, alpha=0.8, s=22)
    ax[1].scatter(s.ratio, s.recall15, c=col[m], alpha=0.8, s=22)
    s2 = s.dropna(subset=["nav_reach_top1pct"])
    ax[2].scatter(s2.ratio, s2.gain_nav, c=col[m], alpha=0.8, s=22)
    ax[3].scatter(s2.ratio, s2.gain_carry, c=col[m], alpha=0.8, s=22)
for a_, t, y in zip(ax, ["fitness kNN Spearman", "recall@15 of walker graph", "nav gain over random walk, top 1% (no memory), pts", "nav gain over random walk, top 1% (memory kept), pts"], [None] * 4):
    a_.set_xscale("log"); a_.set_xlabel("expansion ratio  n_KC / d_input"); a_.set_title(t, fontsize=10)
ax[0].legend(title="ESM-2")
fig.suptitle("FlyHash (random wiring, 5% sparsity), every (d, n_KC) cell, mean over seeds")
fig.tight_layout(); fig.savefig(OUT / "vs_expansion_ratio.png", dpi=130)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
meths = ["dense_cosine", "centered_cosine", "simhash", "flyhash"]
for i, (met, ttl) in enumerate([("fit_rho", "fitness kNN Spearman"), ("recall15", "recall@15 of walker graph"), ("gain_carry", "nav gain over random walk, top 1% (memory kept), pts")]):
    w = 0.25
    for j, m in enumerate(MODELS):
        s = df[df.model == m]
        D = int(s.D.iloc[0])
        vals, errs = [], []
        for me in (meths if met != "gain_carry" else ["dense_cosine", "centered_cosine", "simhash", "flyhash"]):
            r = s[(s.method == me) & (s.d == D) & (s.n_kc.isin([0, BASE_KC])) & (s.sparsity.isna() | (s.sparsity == BASE_SP))][met]
            vals.append(r.mean()); errs.append(r.std(ddof=1))
        ax[i].bar(np.arange(len(vals)) + (j - 1) * w, vals, w, yerr=errs, color=col[m], label=m, capsize=2)
    ax[i].set_xticks(range(len(meths)))
    ax[i].set_xticklabels(meths if met != "gain_carry" else ["dense_cosine", "centered_cosine", "simhash", "flyhash"], rotation=15)
    ax[i].set_title(ttl, fontsize=10)
ax[0].legend(title="ESM-2")
fig.suptitle("Baseline (full dimension, 2000 KC, 5%), mean ± std over seeds")
fig.tight_layout(); fig.savefig(OUT / "baseline_methods.png", dpi=130)

fig, ax = plt.subplots(3, 2, figsize=(10, 11))
for i, m in enumerate(MODELS):
    s = fh[fh.model == m].groupby(["d", "n_kc"]).mean(numeric_only=True).reset_index()
    for j, (met, ttl) in enumerate([("fit_rho", "fitness kNN Spearman"), ("gain_carry", "nav gain over random, top 1% (memory kept), pts")]):
        p = s.pivot(index="d", columns="n_kc", values=met)
        im = ax[i, j].imshow(p.values, aspect="auto", cmap="viridis", origin="lower")
        ax[i, j].set_xticks(range(len(p.columns))); ax[i, j].set_xticklabels(p.columns)
        ax[i, j].set_yticks(range(len(p.index))); ax[i, j].set_yticklabels(p.index)
        ax[i, j].set_title(f"{m}: {ttl}", fontsize=10); ax[i, j].set_xlabel("n_KC"); ax[i, j].set_ylabel("d_input")
        for (y, x), v in np.ndenumerate(p.values):
            if not np.isnan(v):
                ax[i, j].text(x, y, f"{v:.2f}" if met == "fit_rho" else f"{v:.0f}", ha="center", va="center", color="w", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "heatmaps.png", dpi=130)
print((OUT / "tables.md").read_text())
