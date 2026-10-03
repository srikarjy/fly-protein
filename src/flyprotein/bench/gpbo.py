"""Gaussian-process Bayesian optimisation over the discrete candidate set. Deliberately simple and auditable.

Surrogate   exact GP, Matern-5/2 kernel on 50-d PCA of the cached ESM-2 embeddings (PCA is fit on all candidate
            embeddings, which are unlabeled inputs), fit only on the variants queried so far (targets standardised).
            k(r) = (1 + sqrt(5) r/l + 5 r^2 / (3 l^2)) exp(-sqrt(5) r/l);  signal variance 1; Gaussian noise s^2.
Fit         (l, s^2) picked from a fixed grid by log marginal likelihood, re-picked every 20 queries.
            l in {0.5, 1, 2, 4} x (median pairwise PCA distance = 1), s^2 in {1e-3, 1e-2, 1e-1}.
Acquisition Expected Improvement over the best standardised observation, maximised over all unqueried candidates.
Design      the shared start variant plus 9 random variants (10 initial queries, not tuned), then one EI query at a time.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.stats import norm

GRID_L = (0.5, 1.0, 2.0, 4.0)
GRID_S2 = (1e-3, 1e-2, 1e-1)
N_INIT = 10
REFIT_EVERY = 20


def matern52(A, B, ell, a2=None, b2=None):
    a2 = (A * A).sum(1) if a2 is None else a2
    b2 = (B * B).sum(1) if b2 is None else b2
    r = np.sqrt(np.maximum(a2[:, None] + b2[None, :] - 2 * A @ B.T, 0.0)) / ell
    s5 = np.sqrt(5.0) * r
    return (1 + s5 + 5.0 * r * r / 3.0) * np.exp(-s5)


def _lml(Xq, y, ell, s2):
    K = matern52(Xq, Xq, ell) + s2 * np.eye(len(y))
    try:
        c = cho_factor(K, lower=True)
    except np.linalg.LinAlgError:
        return -np.inf
    a = cho_solve(c, y)
    return -0.5 * y @ a - np.log(np.diag(c[0])).sum() - 0.5 * len(y) * np.log(2 * np.pi)


def gpbo(static, oracle, rng, start, trace=None):
    X = static.pca
    n = len(X)
    x2 = (X * X).sum(1)
    idx = [start]
    f0 = oracle.query(start)
    if trace is not None:
        trace.append(dict(type="init", chosen=int(start), q=1, f=f0))
    while len(idx) < N_INIT:
        i = int(rng.integers(n))
        if not oracle.queried(i):
            fi = oracle.query(i)
            idx.append(i)
            if trace is not None:
                trace.append(dict(type="init", chosen=i, q=oracle.n_queries, f=fi))
    hp = None
    while True:
        idx_arr, y = oracle.observed()
        ys = (y - y.mean()) / (y.std() if y.std() > 0 else 1.0)
        Xq = X[idx_arr]
        if hp is None or len(idx_arr) % REFIT_EVERY == 0:
            hp = max(((l, s) for l in GRID_L for s in GRID_S2), key=lambda p: _lml(Xq, ys, *p))
        ell, s2 = hp
        K = matern52(Xq, Xq, ell, x2[idx_arr], x2[idx_arr]) + s2 * np.eye(len(ys))
        c = cho_factor(K, lower=True)
        alpha = cho_solve(c, ys)
        mask = np.ones(n, bool)
        mask[idx_arr] = False
        cand = np.flatnonzero(mask)
        Ks = matern52(X[cand], Xq, ell, x2[cand], x2[idx_arr])
        mu = Ks @ alpha
        v = solve_triangular(c[0], Ks.T, lower=True)
        sd = np.sqrt(np.maximum(1.0 - (v * v).sum(0), 1e-12))
        z = (mu - ys.max()) / sd
        ei = (mu - ys.max()) * norm.cdf(z) + sd * norm.pdf(z)
        pick = int(np.argmax(ei))
        if trace is not None:
            top = np.argsort(-ei)[:5]
            sy, my = (y.std() if y.std() > 0 else 1.0), y.mean()
            tr = dict(type="bo", chosen=int(cand[pick]), ell=float(ell), s2=float(s2), n_obs=len(idx_arr),
                      top=[dict(i=int(cand[t]), ei=float(ei[t]), mu=float(mu[t] * sy + my), sd=float(sd[t] * sy)) for t in top])
        fc = oracle.query(int(cand[pick]))
        if trace is not None:
            tr.update(q=oracle.n_queries, f=float(fc))
            trace.append(tr)
