"""Fixed projection from an embedding (e.g. 320-d ESM-2) down to the PN channel count.

Real hemibrain PN->KC wiring is defined over ~100 PN channels, so embeddings must
be reduced to that width first. A seeded Gaussian random projection is used rather
than PCA so the fly's input layer is not fit to any one assay (Johnson-Lindenstrauss
keeps distances approximately).
"""
from __future__ import annotations

import numpy as np


def gaussian_projection(d_in: int, d_out: int, seed: int = 0) -> np.ndarray:
    """(d_in, d_out) matrix; apply as X @ P."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((d_in, d_out)) / np.sqrt(d_out)).astype(np.float32)
