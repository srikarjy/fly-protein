"""Cross-assay transfer of the fly learner's KC->output weights (leave-one-assay-out).

    python scripts/transfer_experiment.py --seeds 10 --pretrain-starts 100

For each held-out assay h and seed: pre-train the learner (same rule, same FlyHash wiring, 2000 KC, 5%)
on walks over the OTHER assays only, then evaluate on h with the usual 50 starts x 200 steps.
Held-out fitness is never used before evaluation. Rewards are per-assay z-scored fitness (calibration
assumption: the environment reports reward on a standardised scale; reach rates are rank-based so unaffected).
Arms: random | chemotaxis (oracle ref) | zero | zero_mem | pre | pre_mem | pre_frozen | shuf_pre | shuf_pre_mem.
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "prep"))
sys.path.insert(0, str(ROOT / "scripts"))
from build_data import emb_cache_path, knn_cosine  # noqa: E402
from embed_assays import ASSAYS, MODEL  # noqa: E402

from flyprotein.flyhash import FlyHash  # noqa: E402
from flyprotein.navigate import SparseFeat, run_walks  # noqa: E402

N_KC, SP = 2000, 0.05
_cache = {}
REWARD_SCALE = 0.45  # TEM-1's raw fitness std, i.e. the regime the learner's temperature/lr were set for


def protein(a):
    return "_".join(a.split("_")[:2])


def load(a):
    global REWARD_SCALE
    if a not in _cache:
        E = np.load(emb_cache_path(ROOT / "data/cache", a, MODEL)).astype(np.float32)
        f = np.load(ROOT / f"data/cache/{a}.fitness.npy")
        Ec = E - E.mean(0)  # centring uses embeddings only, never fitness
        _cache[a] = dict(Ec=Ec, nbrs=knn_cosine(Ec, 15), z=(f - f.mean()) / f.std() * REWARD_SCALE)
    return _cache[a]


def pretrain(train, codes, seed, starts, shuffle):
    rng = np.random.default_rng(10_000 + seed)
    w = np.zeros(N_KC)
    for k, a in enumerate(rng.permutation(train)):
        z = load(a)["z"]
        if shuffle:
            z = rng.permutation(z)  # keeps the reward distribution, destroys code->fitness
        feat = SparseFeat(codes[a], n_feat=N_KC)
        run_walks(load(a)["nbrs"], z, "learner", feat=feat, carry=True, seed=20_000 + seed * 100 + k, starts=starts, w0=w)
        w = feat.w.copy()
    return w


def job(args):
    seed, starts, scale = args
    global REWARD_SCALE
    REWARD_SCALE = scale  # spawn-safe: workers do not inherit the parent's globals
    _cache.clear()
    fh = FlyHash(d_in=320, n_kc=N_KC, active_frac=SP, seed=seed)  # one wiring per seed, shared by all assays
    codes = {a: np.nonzero(fh.encode(load(a)["Ec"]))[1].reshape(-1, fh.k_active).astype(np.int32) for a in ASSAYS}
    rows = []
    for h in ASSAYS:
        train = [a for a in ASSAYS if protein(a) != protein(h)]
        assert h not in train
        w, w_s = pretrain(train, codes, seed, starts, False), pretrain(train, codes, seed, starts, True)
        L = load(h)
        ev = lambda **kw: run_walks(L["nbrs"], L["z"], seed=seed + 1, **kw)

        def learner(w0=None, carry=False, lr=0.1):
            feat = SparseFeat(codes[h], lr=lr, n_feat=N_KC)
            return ev(mode="learner", feat=feat, carry=carry, w0=w0)

        res = {
            "random": ev(mode="random"), "chemotaxis": ev(mode="chemotaxis"),
            "zero": learner(), "zero_mem": learner(carry=True),
            "pre": learner(w), "pre_mem": learner(w, carry=True), "pre_frozen": learner(w, lr=0.0),
            "shuf_pre": learner(w_s), "shuf_pre_mem": learner(w_s, carry=True),
            # optimistic upper bound: alpha shrinks the prior; chosen post hoc, so NOT a valid result by itself
            "pre_a0.1": learner(0.1 * w), "pre_a0.3": learner(0.3 * w),
            "pre_mem_a0.1": learner(0.1 * w, carry=True), "pre_mem_a0.3": learner(0.3 * w, carry=True),
        }
        zs = {"pre": spearmanr(w[codes[h]].sum(1), L["z"]).statistic, "shuf_pre": spearmanr(w_s[codes[h]].sum(1), L["z"]).statistic}
        for arm, r in res.items():
            rows.append(dict(heldout=h, seed=seed, arm=arm, top1=r["reach_top1pct"] * 100, top10=r["reach_top10pct"] * 100,
                             median_best=r["median_best"], zeroshot_rho=zs.get(arm, np.nan)))
    print(f"seed {seed} done", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--pretrain-starts", type=int, default=100)
    ap.add_argument("--reward-scale", type=float, default=0.45)
    ap.add_argument("--out", default=str(ROOT / "results/transfer/raw.csv"))
    a = ap.parse_args()
    with Pool(3) as p:
        rows = [r for part in p.imap_unordered(job, [(s, a.pretrain_starts, a.reward_scale) for s in range(a.seeds)]) for r in part]
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("wrote", a.out, len(rows))
