"""Search methods. Every method has the signature  method(static, oracle, rng, start, **cfg)  and may learn
about fitness ONLY through oracle.query(). Runs stop when the oracle raises BudgetExhausted (handled by run_method).

What each method is allowed to see
  random_search   : nothing but the candidate count.
  random_walk     : the search graph; fitness of nothing.
  greedy_local    : the search graph and the fitness of variants it has queried (it queries neighbours).
  regevo          : the search graph and fitness of queried variants (population of queried variants).
  fly_*           : the search graph, FlyHash codes of candidates, and the reward (fitness change, scaled by the
                    std of fitness values it has queried) of the variants it moves to. It never queries neighbours.
  zeroshot        : a precomputed protein-language-model score for every candidate, built without any observed fitness
                    measurement (a function of the sequence only); no feedback. The query budget counts the top-ranked
                    candidates subsequently evaluated against the assay, not observations used to train the model.
  gpbo (gpbo.py)  : PCA embeddings of all candidates and fitness of queried variants.
"""
from __future__ import annotations

import numpy as np

from .oracle import BudgetExhausted, Oracle

EPISODE_LEN = 200  # steps per walk before a restart (same for random walk and fly methods)


def _random_unqueried(oracle: Oracle, n: int, rng) -> int:
    for _ in range(1000):
        i = int(rng.integers(n))
        if not oracle.queried(i):
            return i
    return int(rng.integers(n))


def random_search(static, oracle, rng, start):
    for i in rng.permutation(static.n):
        oracle.query(i)


def zeroshot(static, oracle, rng, start, *, scores_key: str):
    order = np.argsort(-static.zs[scores_key], kind="stable")  # best score first; ties by index (deterministic)
    for i in order:
        oracle.query(i)


def random_walk(static, oracle, rng, start):
    n, k = static.nbrs.shape
    pos = start
    for _ in range(100 * oracle.budget):  # step cap: revisits are free, so a walk can loop without using budget
        oracle.query(pos)
        for _ in range(EPISODE_LEN):
            pos = int(static.nbrs[pos, rng.integers(k)])
            oracle.query(pos)
        pos = _random_unqueried(oracle, n, rng)


def greedy_local(static, oracle, rng, start):
    """Steepest-ascent hill climbing with random restarts.
    Query all graph neighbours of the current variant (each new one costs a query), move to the best one if it
    strictly beats the current fitness, otherwise (local optimum) restart at a random unqueried variant."""
    pos, f = start, oracle.query(start)
    while True:
        nb = static.nbrs[pos]
        vals = np.array([oracle.query(j) for j in nb])
        j = int(np.argmax(vals))
        if vals[j] > f:
            pos, f = int(nb[j]), float(vals[j])
        else:
            pos = _random_unqueried(oracle, static.n, rng)
            f = oracle.query(pos)


def regevo(static, oracle, rng, start, *, pop_size=20, sample=5):
    """Regularized evolution (Real et al. 2019) on the search graph: tournament-select the fittest of `sample`
    population members, mutate it by moving to an unqueried graph neighbour, append the child, drop the oldest."""
    pop = [(start, oracle.query(start))]
    while len(pop) < pop_size:
        i = _random_unqueried(oracle, static.n, rng)
        pop.append((i, oracle.query(i)))
    while True:
        cand = [pop[j] for j in rng.choice(len(pop), size=min(sample, len(pop)), replace=False)]
        parent = max(cand, key=lambda t: t[1])[0]
        nb = [int(j) for j in static.nbrs[parent] if not oracle.queried(j)]
        child = int(rng.choice(nb)) if nb else _random_unqueried(oracle, static.n, rng)
        pop.append((child, oracle.query(child)))
        pop.pop(0)


def fly(static, oracle, rng, start, *, codes, T=0.1, lr=0.1, memory=False, replay=0, scale=0.45):
    """The Fly Walker learner (reward-prediction-error rule, unchanged) under query accounting.

    Each move queries only the variant it moves to; that single measurement is the reward. Reward is the change in
    fitness divided by the std of the fitness values queried so far (x `scale`, the regime the learner was set for), so
    no global fitness statistic of the assay is used. memory=False resets the KC weights at every episode; True keeps
    them across episodes (restarts every EPISODE_LEN steps)."""
    n_kc = int(codes.max()) + 1
    ka = codes.shape[1]
    w = np.zeros(n_kc)
    k = static.nbrs.shape[1]
    val = lambda i: w[codes[i]].sum(-1)

    def std_obs():
        _, y = oracle.observed()
        return y.std() if len(y) > 1 else 0.0

    def update(cur, nxt, df):
        s = std_obs()
        if s <= 0 or df == 0:
            return
        delta = df * scale / s - (float(val(nxt)) - float(val(cur)))
        a = lr * delta / (2 * ka)
        w[codes[nxt]] += a
        w[codes[cur]] -= a

    pos = start
    first = True
    for _ in range(100 * oracle.budget):
        if not first:
            pos = _random_unqueried(oracle, static.n, rng)
            if not memory:
                w[:] = 0.0
        first = False
        oracle.query(pos)
        buf = []
        for _ in range(EPISODE_LEN):
            nb = static.nbrs[pos]
            vals = val(nb)
            p = np.exp((vals - vals.max()) / T)
            j = min(int(np.searchsorted(np.cumsum(p), rng.random() * p.sum())), k - 1)
            nxt = int(nb[j])
            f_cur = oracle.query(pos)
            df = oracle.query(nxt) - f_cur
            update(pos, nxt, df)
            if replay:
                buf.append((pos, nxt, df))
                for r in rng.integers(0, len(buf), size=replay):
                    c_, n_, d_ = buf[r]
                    update(c_, n_, d_)
            pos = nxt


METHODS = {
    "random_search": random_search,
    "random_walk": random_walk,
    "greedy_local": greedy_local,
    "regevo": regevo,
}


def run_method(fn, static, fitness, budget, start, rng, **cfg):
    """Run `fn` until the oracle budget is spent; returns the ordered query log (the deterministic record)."""
    oracle = Oracle(fitness, budget)
    try:
        fn(static, oracle, rng, start, **cfg)
    except BudgetExhausted:
        pass
    return oracle.log

from .gpbo import gpbo  # noqa: E402

METHODS["gpbo_ei"] = gpbo

from .adalead import adalead_graph  # noqa: E402

METHODS["adalead_graph"] = adalead_graph
