"""Per-assay static inputs. `Static` holds everything a method may use: sequences, embeddings, the
search graph and FlyHash codes. It contains NO fitness values."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from flyprotein.flyhash import FlyHash

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data/cache"
MODEL = "esm2_t6_8M_UR50D"  # embedding used by every embedding-based method (cached for all benchmark assays)


@dataclass
class Static:
    name: str
    n: int
    wt: str
    mutants: list
    emb: np.ndarray  # (n, d) raw ESM-2 embeddings
    nbrs: np.ndarray  # (n, k) embedding kNN graph (cosine, mean-centred), the same graph for every local method
    pca: np.ndarray  # (n, p) PCA of centred embeddings, scaled so the median pairwise distance is 1 (GP-BO input)
    zs: dict = field(default_factory=dict)  # zero-shot scores by name (no fitness involved)

    def codes(self, seed: int, n_kc: int = 2000, sparsity: float = 0.05) -> np.ndarray:
        """FlyHash codes (n, k_active); wiring depends on `seed`."""
        Ec = self.emb - self.emb.mean(0)
        fh = FlyHash(d_in=self.emb.shape[1], n_kc=n_kc, active_frac=sparsity, seed=seed)
        return np.nonzero(fh.encode(Ec))[1].reshape(self.n, fh.k_active).astype(np.int32)


def _knn(Ec: np.ndarray, k: int) -> np.ndarray:
    Z = Ec / np.linalg.norm(Ec, axis=1, keepdims=True).clip(min=1e-9)
    out = np.empty((len(Z), k), dtype=np.int32)
    for i in range(0, len(Z), 1024):
        sim = Z[i : i + 1024] @ Z.T
        sim[np.arange(sim.shape[0]), np.arange(i, i + sim.shape[0])] = -np.inf
        idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sim, idx, axis=1), axis=1)
        out[i : i + 1024] = np.take_along_axis(idx, order, axis=1)
    return out


def make_static(name: str, k: int = 15, pca_dim: int = 50) -> Static:
    meta = json.load(open(CACHE / f"{name}.variants.json"))
    E = np.load(CACHE / f"{name}.{MODEL}.emb.npy").astype(np.float32)
    Ec = E - E.mean(0)
    _, _, Vt = np.linalg.svd(Ec, full_matrices=False)
    P = Ec @ Vt[:pca_dim].T
    sub = np.random.default_rng(0).choice(len(P), size=min(1000, len(P)), replace=False)
    d = np.linalg.norm(P[sub][:, None] - P[sub][None], axis=-1)
    P = (P / np.median(d[d > 0])).astype(np.float64)
    zs = {p.name.split(".zs.")[1][:-4]: np.load(p) for p in sorted(CACHE.glob(f"{name}.zs.*.npy"))}
    return Static(name, len(E), meta["wt"], meta["mutants"], E, _knn(Ec, k), P, zs)


def load_assay_data(name: str) -> np.ndarray:
    """Measured fitness. Only the runner (to build an Oracle) and the evaluator may call this."""
    return np.load(CACHE / f"{name}.fitness.npy")
