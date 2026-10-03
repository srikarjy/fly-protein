"""AdaLead (Sinai et al. 2020), adapted to a measured candidate set. NOT the published code: read this before citing.

Kept from AdaLead:   seeds = queried variants within kappa of the best observed fitness (kappa = 0.05, the FLEXS default);
                     from each seed, model-guided hill climbing: propose a random mutation, accept it iff the surrogate
                     predicts fitness >= the parent's; accepted children become batch proposals; the top proposals by
                     surrogate score are measured for real (batch), then the surrogate is refit on everything measured.
Adapted (and why):   only variants in the assay can be measured, so a "mutation" is a move to a random neighbour in the
                     embedding kNN graph (the same graph the local methods use); recombination is dropped (no
                     sequence-level oracle for double mutants); the surrogate is the GP posterior mean of gpbo.py;
                     kappa is relative to the observed range, (f - f_min)/(f_max - f_min) >= 1 - kappa;
                     batch = 10 real queries per round; up to 20 hill-climb proposals per seed (not tuned).
If fewer than `batch` proposals are new, the batch is filled with the highest-scoring unqueried candidates.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .gpbo import GRID_L, GRID_S2, N_INIT, REFIT_EVERY, _lml, matern52

KAPPA, BATCH, CLIMB = 0.05, 10, 20


def adalead_graph(static, oracle, rng, start):
    X = static.pca
    n = len(X)
    x2 = (X * X).sum(1)
    oracle.query(start)
    while oracle.n_queries < N_INIT:
        i = int(rng.integers(n))
        oracle.query(i)
    hp = None
    refit_at = -1
    while True:
        idx, y = oracle.observed()
        ys = (y - y.mean()) / (y.std() if y.std() > 0 else 1.0)
        Xq = X[idx]
        if hp is None or (len(idx) // REFIT_EVERY) != refit_at:
            hp = max(((l, s) for l in GRID_L for s in GRID_S2), key=lambda p: _lml(Xq, ys, *p))
            refit_at = len(idx) // REFIT_EVERY
        ell, s2 = hp
        c = cho_factor(matern52(Xq, Xq, ell, x2[idx], x2[idx]) + s2 * np.eye(len(ys)), lower=True)
        mu = matern52(X, Xq, ell, x2, x2[idx]) @ cho_solve(c, ys)  # surrogate prediction for every candidate
        span = ys.max() - ys.min()
        seeds = idx[(ys - ys.min()) >= (1 - KAPPA) * span] if span > 0 else idx
        proposals = {}
        for s in seeds:
            cur = int(s)
            for _ in range(CLIMB):
                child = int(static.nbrs[cur, rng.integers(static.nbrs.shape[1])])
                if mu[child] >= mu[cur]:
                    cur = child
                    if not oracle.queried(child):
                        proposals[child] = mu[child]
        order = sorted(proposals, key=lambda i: -proposals[i])[:BATCH]
        if len(order) < BATCH:
            rest = np.argsort(-mu)
            for i in rest:
                if len(order) >= BATCH:
                    break
                if not oracle.queried(i) and int(i) not in order:
                    order.append(int(i))
        for i in order:
            oracle.query(i)
