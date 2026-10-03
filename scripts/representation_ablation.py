"""Representation/compression ablation for the fly-inspired projection (no learner changes).

    python scripts/representation_ablation.py --procs 3          # full grid -> results/ablation/raw.csv

Per (embedding model, seed) it sweeps input (PN) dimension d, KC count and sparsity and
reports, for dense cosine, centered cosine, SimHash and FlyHash (random wiring):

  a) representation:  recall@15 of the walker's kNN graph (centered-cosine kNN in the full
     embedding) and Spearman of the leave-one-out kNN fitness prediction;
  b) navigation:      the web app's walker (50 starts x 200 steps) reach rates and median best
     fitness, with the reward-prediction-error rule unchanged (flyprotein.navigate).

Uses cached embeddings in data/cache (no GPU, no regeneration).
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "prep"))
from build_data import knn_cosine  # noqa: E402

from flyprotein.flyhash import FlyHash  # noqa: E402
from flyprotein.navigate import DenseFeat, SparseFeat, run_walks  # noqa: E402
from flyprotein.project import gaussian_projection  # noqa: E402

MODELS = {"8M": "esm2_t6_8M_UR50D", "150M": "esm2_t30_150M_UR50D", "650M": "esm2_t33_650M_UR50D"}
DIMS = [64, 128, 256, 512, 1024]
KCS = [2000, 4000, 8000, 16000]
SPARS = [0.02, 0.05, 0.10]
BASE_KC, BASE_SP = 2000, 0.05
K = 15
ASSAY = "BLAT_ECOLX_Firnberg_2014"


def retrieval(S, q, ref, fit, rng, ties=False):
    S = S.astype(np.float32, copy=True)
    if ties:  # integer similarities: random tie-breaking so index order can't leak in
        S += rng.random(S.shape, dtype=np.float32) * 1e-3
    S[np.arange(len(q)), q] = -np.inf
    top = np.argpartition(-S, K, axis=1)[:, :K]
    recall = (top[:, :, None] == ref[q][:, None, :]).any(-1).sum(1).mean() / K
    rho = spearmanr(fit[top].mean(1), fit[q]).statistic
    return float(recall), float(rho)


def nav(nbrs, fit, seed, feat=None, mode="learner", carry=False):
    return run_walks(nbrs, fit, mode, feat=feat, carry=carry, seed=seed + 1)


def job(args):
    model, seed, quick = args
    t0 = time.time()
    fit = np.array([v["f"] for v in json.load(open(ROOT / "web/public/data.json"))["variants"]])
    E = np.load(ROOT / f"data/cache/{ASSAY}.{MODELS[model]}.emb.npy").astype(np.float32)
    n, D = E.shape
    Ec = E - E.mean(0)
    ref = knn_cosine(Ec, K)
    q = np.random.default_rng(0).choice(n, 1000, replace=False)
    rng = np.random.default_rng(seed)
    dims = [d for d in DIMS if d < D] + [D]
    kcs = KCS[:2] if quick else KCS
    rows = []

    def add(method, d, n_kc, sp, rec, rho, navs=None):
        r = dict(model=model, D=D, seed=seed, method=method, d=d, n_kc=n_kc, sparsity=sp,
                 ratio=(n_kc / d) if n_kc else np.nan, recall15=rec, fit_rho=rho)
        for tag, res in (navs or {}).items():
            r.update({f"{tag}_{k}": v for k, v in res.items()})
        rows.append(r)

    # references independent of representation
    add("random_walk", D, 0, np.nan, np.nan, np.nan, {"nav": nav(ref, fit, seed, mode="random")})  # same graph, same starts as every learner
    add("chemotaxis", D, 0, np.nan, np.nan, np.nan, {"nav": nav(ref, fit, seed, mode="chemotaxis")})

    for d in dims:
        P = None if d == D else gaussian_projection(D, d, seed)
        Xc = Ec if P is None else Ec @ P
        Xr = E if P is None else E @ P
        for name, X in (("dense_cosine", Xr), ("centered_cosine", Xc)):
            phi = (X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-9)).astype(np.float32)
            rec, rho = retrieval(phi[q] @ phi.T, q, ref, fit, rng)
            feat = DenseFeat(phi)
            add(name, d, 0, np.nan, rec, rho, {"nav": nav(ref, fit, seed, feat), "navcarry": nav(ref, fit, seed, feat, carry=True)})
        for n_kc in kcs:
            R = np.random.default_rng(seed + 7).standard_normal((d, n_kc)).astype(np.float32)
            B = np.where(Xc @ R > 0, 1.0, -1.0).astype(np.float32)
            rec, rho = retrieval(B[q] @ B.T, q, ref, fit, rng, ties=True)
            feat = DenseFeat(B)
            add("simhash", d, n_kc, np.nan, rec, rho, {"nav": nav(ref, fit, seed, feat), "navcarry": nav(ref, fit, seed, feat, carry=True)})
            del R, B
            for sp in SPARS:
                fh = FlyHash(d_in=d, n_kc=n_kc, active_frac=sp, seed=seed).fit(Xc)
                C = fh.encode(Xc)
                codes = np.nonzero(C)[1].reshape(n, fh.k_active).astype(np.int32)
                Cq = C[q].astype(np.float32)
                rec, rho = retrieval(Cq @ C.astype(np.float32).T, q, ref, fit, rng, ties=True)
                navs = None
                if sp == BASE_SP or (d == D and n_kc == BASE_KC):
                    feat = SparseFeat(codes)
                    feat.reset(n_kc)
                    navs = {"nav": nav(ref, fit, seed, feat), "navcarry": nav(ref, fit, seed, feat, carry=True)}
                add("flyhash", d, n_kc, sp, rec, rho, navs)
                del C, Cq, codes
    print(f"done {model} seed {seed}: {len(rows)} rows in {time.time() - t0:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="smoke test: KC 2000/4000 only")
    ap.add_argument("--out", default=str(ROOT / "results/ablation/raw.csv"))
    a = ap.parse_args()
    jobs = [(m, s, a.quick) for s in range(a.seeds) for m in a.models]
    with Pool(a.procs) as p:
        rows = [r for part in p.imap_unordered(job, jobs) for r in part]
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("wrote", a.out, len(rows), "rows")


if __name__ == "__main__":
    main()
