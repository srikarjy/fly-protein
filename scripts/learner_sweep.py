"""Learner variants on the FlyHash baseline cell (full d, 2000 KC, 5%, random wiring).

    python scripts/learner_sweep.py tune    # seeds 100-104: choose a config
    python scripts/learner_sweep.py final   # seeds 0-9: baseline vs chosen configs (paired)

Metric: reach-top-1% gain over a random walk on the same graph and starts (points).
"""
import json
import sys
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "prep"))
from build_data import knn_cosine  # noqa: E402

from flyprotein.flyhash import FlyHash  # noqa: E402
from flyprotein.navigate import SparseFeat, run_walks  # noqa: E402

MODELS = {"8M": "esm2_t6_8M_UR50D", "150M": "esm2_t30_150M_UR50D", "650M": "esm2_t33_650M_UR50D"}
ASSAY = "BLAT_ECOLX_Firnberg_2014"
OUT = ROOT / "results/learner"

CONFIGS = {"baseline": dict(rule="difference", lr=0.1, T=0.1, replay=0)}
for lr in (0.03, 0.3, 1.0):
    CONFIGS[f"lr{lr}"] = dict(rule="difference", lr=lr, T=0.1, replay=0)
for T in (0.03, 0.3):
    CONFIGS[f"T{T}"] = dict(rule="difference", lr=0.1, T=T, replay=0)
for lr in (0.1, 0.3, 1.0):
    CONFIGS[f"abs_lr{lr}"] = dict(rule="absolute", lr=lr, T=0.1, replay=0)
for rp in (2, 5, 10):
    CONFIGS[f"replay{rp}"] = dict(rule="difference", lr=0.1, T=0.1, replay=rp)
CONFIGS["replay5_lr0.3"] = dict(rule="difference", lr=0.3, T=0.1, replay=5)
CONFIGS["replay5_T0.3"] = dict(rule="difference", lr=0.1, T=0.3, replay=5)


def job(args):
    model, seed, names = args
    fit = np.array([v["f"] for v in json.load(open(ROOT / "web/public/data.json"))["variants"]])
    E = np.load(ROOT / f"data/cache/{ASSAY}.{MODELS[model]}.emb.npy").astype(np.float32)
    n, D = E.shape
    Ec = E - E.mean(0)
    ref = knn_cosine(Ec, 15)
    fh = FlyHash(d_in=D, n_kc=2000, active_frac=0.05, seed=seed).fit(Ec)
    codes = np.nonzero(fh.encode(Ec))[1].reshape(n, fh.k_active).astype(np.int32)
    rw = run_walks(ref, fit, "random", seed=seed + 1)["reach_top1pct"] * 100
    rows = []
    for name in names:
        c = CONFIGS[name]
        for carry in (False, True):
            feat = SparseFeat(codes, lr=c["lr"])
            feat.reset(2000)
            r = run_walks(ref, fit, "learner", feat=feat, carry=carry, seed=seed + 1, temperature=c["T"], rule=c["rule"], replay=c["replay"])
            rows.append(dict(model=model, seed=seed, config=name, carry=carry, top1=r["reach_top1pct"] * 100, rw=rw, gain=r["reach_top1pct"] * 100 - rw, median_best=r["median_best"]))
    return rows


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "tune":
        seeds, names, out = range(100, 105), list(CONFIGS), OUT / "tune.csv"
    else:
        names = sys.argv[2].split(",") if len(sys.argv) > 2 else ["baseline"]
        seeds, out = range(10), OUT / "final.csv"
        names = ["baseline"] + [x for x in names if x != "baseline"]
    jobs = [(m, s, names) for s in seeds for m in MODELS]
    with Pool(3) as p:
        rows = [r for part in p.imap_unordered(job, jobs) for r in part]
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, len(rows))
