"""Python port of the Fly Walker agents (web/src/fly/*.ts) for offline ablations.

The learning rule is the web app's reward-prediction-error rule, unchanged:

    delta = (fitness[next] - fitness[cur]) - (value(next) - value(cur))
    w    += lr * delta * (phi[next] - phi[cur]) / (|phi[next]|^2 + |phi[cur]|^2)

For binary KC codes |phi|^2 is the number of active cells and this is exactly the
JS update (w[next cells] += a, w[cur cells] -= a, a = lr*delta/(n_next+n_cur)).
Writing it over a generic feature vector phi lets dense cosine and SimHash features
use the same rule, so the representation is the only thing that changes.
"""
from __future__ import annotations

import numpy as np


class SparseFeat:
    """Binary KC codes, fixed number of active cells per variant. codes: (n, k_active)."""

    def __init__(self, codes: np.ndarray, lr: float = 0.1, n_feat: int | None = None):
        self.codes = codes
        self.ka = codes.shape[1]
        self.lr = lr
        self.n_feat = n_feat or int(codes.max()) + 1
        self.reset()

    def reset(self, n_feat: int | None = None):
        self.w = np.zeros(n_feat or self.n_feat, dtype=np.float64)

    def value(self, idx):
        return self.w[self.codes[idx]].sum(-1)

    def update(self, cur: int, nxt: int, delta: float):
        a = self.lr * delta / (2 * self.ka)
        self.w[self.codes[nxt]] += a
        self.w[self.codes[cur]] -= a


    def update_abs(self, nxt: int, delta: float):
        """Absolute rule: learn each visited variant's fitness (delta = fitness - value)."""
        self.w[self.codes[nxt]] += self.lr * delta / self.ka


class DenseFeat:
    """Dense feature rows phi (n, m): normalised embeddings, or +-1 SimHash bits."""

    def __init__(self, phi: np.ndarray, lr: float = 0.1):
        self.phi = phi.astype(np.float32, copy=False)
        self.norm2 = (self.phi.astype(np.float64) ** 2).sum(1)
        self.lr = lr
        self.reset()

    def reset(self, n_feat: int | None = None):
        self.w = np.zeros(self.phi.shape[1], dtype=np.float64)

    def value(self, idx):
        return self.phi[idx] @ self.w.astype(np.float32)

    def update(self, cur: int, nxt: int, delta: float):
        a = self.lr * delta / (self.norm2[nxt] + self.norm2[cur])
        self.w += a * (self.phi[nxt] - self.phi[cur])

    def update_abs(self, nxt: int, delta: float):
        self.w += self.lr * delta / self.norm2[nxt] * self.phi[nxt]


def _softmax_pick(vals: np.ndarray, temperature: float, u: float) -> int:
    p = np.exp((vals - vals.max()) / temperature)
    c = np.cumsum(p)
    j = int(np.searchsorted(c, u * c[-1]))
    return min(j, len(vals) - 1)


def run_walks(
    nbrs: np.ndarray,
    fitness: np.ndarray,
    mode: str,
    feat=None,
    carry: bool = False,
    seed: int = 1,
    starts: int = 50,
    steps: int = 200,
    temperature: float = 0.1,
    top_fracs=(0.1, 0.01),
    rule: str = "difference",
    replay: int = 0,
    w0: np.ndarray | None = None,
) -> dict:
    """Same starts for every mode at a given seed. mode: 'random' | 'chemotaxis' | 'learner'.

    w0: initial KC->output weights (default zeros); with carry=False every episode restarts from w0.
    rule: 'difference' (web app default) or 'absolute'. replay: extra updates per step on
    transitions already experienced in this episode, recomputed with the current weights
    (the fly never sees a fitness it has not visited). Defaults reproduce the web app.

    Returns median best fitness and, per top fraction, the share of starts that
    reached that top fraction of the landscape within the step budget.
    """
    n = len(fitness)
    start_rng = np.random.default_rng(seed)
    start_list = start_rng.integers(0, n, size=starts)
    rng = np.random.default_rng(seed + 1000)
    thr = {f: np.quantile(fitness, 1 - f) for f in top_fracs}
    bests, reached = [], {f: 0 for f in top_fracs}

    def init_weights():
        feat.reset()
        if w0 is not None:  # start from pre-trained synapses instead of zero
            feat.w[:] = w0

    if feat is not None and carry:
        init_weights()
    for s in start_list:
        if feat is not None and not carry:
            init_weights()
        pos = int(s)
        buf = []  # transitions experienced this episode
        best = fitness[pos]
        hit = {f: fitness[pos] >= thr[f] for f in top_fracs}
        for _ in range(steps):
            nb = nbrs[pos]
            u = rng.random()
            if mode == "random":
                j = min(int(u * len(nb)), len(nb) - 1)
            elif mode == "chemotaxis":
                j = _softmax_pick(fitness[nb], temperature, u)
            else:
                vals = feat.value(nb)
                j = _softmax_pick(vals, temperature, u)
                nxt = int(nb[j])
                if rule == "absolute":
                    feat.update_abs(nxt, fitness[nxt] - float(vals[j]))
                else:
                    delta = (fitness[nxt] - fitness[pos]) - (float(vals[j]) - float(feat.value(pos)))
                    feat.update(pos, nxt, delta)
                    if replay:
                        buf.append((pos, nxt))
                        for r in rng.integers(0, len(buf), size=replay):
                            c_, n_ = buf[r]
                            d_ = (fitness[n_] - fitness[c_]) - float(feat.value(n_) - feat.value(c_))
                            feat.update(c_, n_, d_)
            pos = int(nb[j])
            f = fitness[pos]
            if f > best:
                best = f
            for fr in top_fracs:
                if not hit[fr] and f >= thr[fr]:
                    hit[fr] = True
        bests.append(best)
        for fr in top_fracs:
            reached[fr] += hit[fr]
    out = {"median_best": float(np.median(bests))}
    for fr in top_fracs:
        out[f"reach_top{fr * 100:g}pct"] = reached[fr] / starts
    return out
