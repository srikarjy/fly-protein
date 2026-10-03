"""Real hemibrain PN->KC wiring vs random wiring, reproducible Python version of the earlier browser evaluation.

    python scripts/hemibrain_ablation.py
TEM-1 (BLAT_ECOLX_Firnberg_2014) with ESM-2 8M/150M/650M embeddings, 10 seeds. Arms (all 5% sparsity, FlyHash readout):
  random_2000  random wiring, 2000 KC (the default used elsewhere)
  random_1785  random wiring, 1785 KC (same code length as the hemibrain circuit)
  hemibrain    hemibrain:v1.2.1 PN->KC synapse counts (135 PNs -> 1,785 KCs); embedding reduced to 135 channels by a fixed
               seeded Gaussian projection.
Metrics: kNN fitness-prediction Spearman and recall@15 of the walker graph (representation); navigation = reach-top-1% gain over
a random walk, 50 starts x 200 steps, without / with memory kept across starts (flyprotein.navigate.run_walks, rule unchanged).
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "prep"))
sys.path.insert(0, str(ROOT / "scripts"))
from build_data import knn_cosine  # noqa: E402
from representation_ablation import retrieval  # noqa: E402

from flyprotein.connectome import load_pn_kc_edges  # noqa: E402
from flyprotein.flyhash import FlyHash  # noqa: E402
from flyprotein.navigate import SparseFeat, run_walks  # noqa: E402
from flyprotein.project import gaussian_projection  # noqa: E402

ASSAY = "BLAT_ECOLX_Firnberg_2014"
MODELS = {"8M": "esm2_t6_8M_UR50D", "150M": "esm2_t30_150M_UR50D", "650M": "esm2_t33_650M_UR50D"}


def job(args):
    model, seed = args
    fit = np.load(ROOT / f"data/cache/{ASSAY}.fitness.npy")
    E = np.load(ROOT / f"data/cache/{ASSAY}.{MODELS[model]}.emb.npy").astype(np.float32)
    n, D = E.shape
    Ec = E - E.mean(0)
    ref = knn_cosine(Ec, 15)
    q = np.random.default_rng(0).choice(n, 1000, replace=False)
    rng = np.random.default_rng(seed)
    edges, n_pn, n_kc = load_pn_kc_edges(cache=str(ROOT / "data/cache/hemibrain_pn_kc.npz"))
    arms = {}
    for name, kc in (("random_2000", 2000), ("random_1785", 1785)):
        fh = FlyHash(d_in=D, n_kc=kc, active_frac=0.05, seed=seed)
        arms[name] = (fh.encode(Ec), kc)
    X = Ec @ gaussian_projection(D, n_pn, seed=seed)
    arms["hemibrain"] = (FlyHash.from_edges(edges, n_pn, n_kc).fit(X).encode(X), n_kc)
    rw = run_walks(ref, fit, "random", seed=seed + 1)["reach_top1pct"] * 100
    rows = []
    for name, (C, kc) in arms.items():
        ka = int(C[0].sum())
        codes = np.nonzero(C)[1].reshape(n, ka).astype(np.int32)
        Cf = C.astype(np.float32)
        rec, rho = retrieval(Cf[q] @ Cf.T, q, ref, fit, rng, ties=True)
        r = dict(model=model, seed=seed, arm=name, n_kc=kc, recall15=rec, fit_rho=rho)
        for carry in (False, True):
            feat = SparseFeat(codes, n_feat=kc)
            res = run_walks(ref, fit, "learner", feat=feat, carry=carry, seed=seed + 1)
            r["gain_carry" if carry else "gain"] = res["reach_top1pct"] * 100 - rw
        rows.append(r)
    return rows


if __name__ == "__main__":
    with Pool(3) as p:
        rows = [r for part in p.imap_unordered(job, [(m, s) for m in MODELS for s in range(10)]) for r in part]
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results/hemibrain/raw.csv", index=False)
    g = d.groupby(["model", "arm"], sort=False)[["fit_rho", "recall15", "gain", "gain_carry"]].agg(["mean", "std"]).round(3)
    print(g.to_string())
    for m in MODELS:
        for col in ("fit_rho", "gain", "gain_carry"):
            a = d[(d.model == m) & (d.arm == "hemibrain")].set_index("seed")[col]
            b = d[(d.model == m) & (d.arm == "random_1785")].set_index("seed")[col]
            print(f"{m} {col}: hemibrain - random_1785 = {(a - b).mean():+.3f} +- {(a - b).std(ddof=1) / np.sqrt(len(a)):.3f} (SE)")
