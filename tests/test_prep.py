import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prep"))
from build_data import hemibrain_codes, knn_cosine  # noqa: E402

from flyprotein.flyhash import FlyHash  # noqa: E402
from flyprotein.project import gaussian_projection  # noqa: E402


def test_knn_cosine_excludes_self_and_finds_neighbors():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((3, 16))
    X = np.vstack([base[i] + 0.01 * rng.standard_normal((5, 16)) for i in range(3)])  # 3 tight clusters of 5
    nbrs = knn_cosine(X, 4)
    assert nbrs.shape == (15, 4)
    for i in range(15):
        assert i not in nbrs[i]
        assert set(nbrs[i]) == set(range(i // 5 * 5, i // 5 * 5 + 5)) - {i}


def test_projection_roughly_preserves_cosine_geometry():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 320)).astype(np.float32)
    Z = X @ gaussian_projection(320, 100, seed=3)
    assert Z.shape == (50, 100)
    assert np.allclose(gaussian_projection(320, 100, seed=3), gaussian_projection(320, 100, seed=3))
    a, b = X[0], X[1]
    cos = lambda u, v: float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))  # noqa: E731
    assert abs(cos(Z[0], Z[1]) - cos(a, b)) < 0.3


def test_flyhash_from_toy_edges_has_exact_sparsity():
    edges = [(pn, kc, 1.0) for kc in range(40) for pn in ((kc * 3) % 10, (kc * 7 + 1) % 10)]
    fh = FlyHash.from_edges(edges, n_pn=10, n_kc=40, active_frac=0.1)
    codes = fh.encode(np.random.default_rng(2).standard_normal((6, 10)))
    assert (codes.sum(axis=1) == 4).all()


def test_hemibrain_codes_sparse_and_deterministic():
    rng = np.random.default_rng(2)
    E = rng.standard_normal((20, 32)).astype(np.float32)
    n_pn, n_kc = 8, 40
    edges = [(int(rng.integers(n_pn)), int(rng.integers(n_kc)), float(rng.integers(1, 9))) for _ in range(120)]
    c1, k = hemibrain_codes(E, edges, n_pn, n_kc, seed=1)
    c2, _ = hemibrain_codes(E, edges, n_pn, n_kc, seed=1)
    assert c1.shape == (20, n_kc) and k == 2
    assert (c1.sum(axis=1) == k).all()
    assert (c1 == c2).all()
