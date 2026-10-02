from __future__ import annotations

import numpy as np


class FingerprintIndex:
    """Nearest-neighbor search over sparse binary fingerprints (overlap count)."""

    def __init__(self, fingerprints: np.ndarray, labels: list | None = None):
        self.fp = fingerprints.astype(np.float32)
        self.labels = labels if labels is not None else list(range(len(fingerprints)))

    def query(self, fp: np.ndarray, k: int = 5):
        """Return [(label, overlap)] for the k best matches per query row."""
        overlap = np.atleast_2d(fp).astype(np.float32) @ self.fp.T
        order = np.argsort(-overlap, axis=1)[:, :k]
        return [[(self.labels[j], int(overlap[i, j])) for j in row] for i, row in enumerate(order)]
