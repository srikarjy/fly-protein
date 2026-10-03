import numpy as np

from flyprotein.navigate import DenseFeat, SparseFeat, run_walks


def ring(n=40, k=2):
    fit = 1 - np.abs(np.arange(n) - 20) / 20
    nbrs = np.stack([(np.arange(n) - 1) % n, (np.arange(n) + 1) % n], 1)
    return nbrs, fit


def test_dense_rule_equals_sparse_rule_on_binary_codes():
    rng = np.random.default_rng(0)
    n, m, ka = 30, 50, 5
    codes = np.stack([rng.choice(m, ka, replace=False) for _ in range(n)])
    phi = np.zeros((n, m), dtype=np.float32)
    np.put_along_axis(phi, codes, 1.0, axis=1)
    s, d = SparseFeat(codes, lr=0.1), DenseFeat(phi, lr=0.1)
    s.reset(m)
    for _ in range(200):
        cur, nxt = rng.integers(n, size=2)
        delta = rng.normal()
        s.update(int(cur), int(nxt), delta)
        d.update(int(cur), int(nxt), delta)
    assert np.allclose(s.w, d.w, atol=1e-6)
    assert np.allclose(s.value(np.arange(n)), d.value(np.arange(n)), atol=1e-5)


def test_chemotaxis_beats_random_and_learner_learns():
    nbrs, fit = ring()
    rnd = run_walks(nbrs, fit, "random", seed=3, starts=30, steps=60)
    chemo = run_walks(nbrs, fit, "chemotaxis", seed=3, starts=30, steps=60)
    assert chemo["reach_top10pct"] >= rnd["reach_top10pct"]
    assert chemo["reach_top10pct"] == 1.0
    # learner whose codes encode position: each variant has its own 2 cells
    codes = np.stack([[2 * i, 2 * i + 1] for i in range(40)])
    f = SparseFeat(codes)
    run_walks(nbrs, fit, "learner", feat=f, carry=True, seed=3, starts=10, steps=60)
    assert np.abs(f.w).sum() > 0
