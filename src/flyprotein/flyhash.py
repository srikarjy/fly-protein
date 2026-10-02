"""FlyHash: the fly olfactory expansion circuit as a similarity hash.

Antennal-lobe projection neurons (PNs, ~50 channels) feed Kenyon cells
(KCs, ~2000) through sparse, near-random wiring. Each KC samples a handful
of PNs, and only the most active few percent of KCs fire (winner-take-all,
enforced by the APL inhibitory neuron). The output is a sparse binary code
where similar inputs share many active KCs.

The projection matrix is either random-sparse (default) or built from real
PN->KC synapse counts via `FlyHash.from_edges`.
"""
from __future__ import annotations

import numpy as np


class FlyHash:
    def __init__(
        self,
        d_in: int,
        n_kc: int = 2000,
        sample_frac: float = 0.1,
        active_frac: float = 0.05,
        seed: int = 0,
        weights: np.ndarray | None = None,
    ):
        self.d_in = d_in
        self.n_kc = n_kc
        self.k_active = max(1, int(round(active_frac * n_kc)))
        if weights is not None:
            if weights.shape != (n_kc, d_in):
                raise ValueError(f"weights must be {(n_kc, d_in)}, got {weights.shape}")
            self.W = weights.astype(np.float32)
        else:
            rng = np.random.default_rng(seed)
            per_kc = max(1, int(round(sample_frac * d_in)))
            W = np.zeros((n_kc, d_in), dtype=np.float32)
            for i in range(n_kc):
                W[i, rng.choice(d_in, size=per_kc, replace=False)] = 1.0
            self.W = W
        self.mean_ = np.zeros(d_in, dtype=np.float32)

    @classmethod
    def from_edges(cls, edges, n_pn: int, n_kc: int, active_frac: float = 0.05):
        """Build from (pn_index, kc_index, synapse_count) triples.

        This is how real hemibrain PN->KC counts become the projection.
        """
        W = np.zeros((n_kc, n_pn), dtype=np.float32)
        for pn, kc, w in edges:
            W[int(kc), int(pn)] += float(w)
        return cls(d_in=n_pn, n_kc=n_kc, active_frac=active_frac, weights=W)

    def fit(self, X: np.ndarray) -> "FlyHash":
        """Learn the input mean (the fly's adaptation to background odor)."""
        self.mean_ = np.asarray(X, dtype=np.float32).mean(axis=0)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Return a boolean (n, n_kc) array with exactly k_active ones per row."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float32)) - self.mean_
        act = X @ self.W.T
        top = np.argpartition(-act, self.k_active - 1, axis=1)[:, : self.k_active]
        out = np.zeros(act.shape, dtype=bool)
        np.put_along_axis(out, top, True, axis=1)
        return out
