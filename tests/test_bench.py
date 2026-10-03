import numpy as np
import pytest

from flyprotein.bench import BudgetExhausted, Oracle, Static, curve_percentiles
from flyprotein.bench.methods import METHODS, fly, run_method, zeroshot


def synth(n=400, seed=0):
    """A smooth 1-D ridge landscape on a ring: neighbours are +-1..7 positions, embeddings vary smoothly."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / n * 2 * np.pi
    emb = np.stack([np.sin(k * t + p) for k in (1, 2, 3, 4) for p in (0, 1, 2, 3)], 1).astype(np.float32)
    emb += 0.01 * rng.standard_normal(emb.shape).astype(np.float32)
    nbrs = np.stack([(np.arange(n) + d) % n for d in list(range(-7, 0)) + list(range(1, 8))] , 1)[:, :15].astype(np.int32)
    fitness = np.exp(-((t - 2.0) ** 2) / 0.1) + 0.3 * np.exp(-((t - 4.5) ** 2) / 0.3)
    pca = (emb @ np.linalg.svd(emb - emb.mean(0), full_matrices=False)[2][:5].T).astype(np.float64)
    pca /= np.median(np.linalg.norm(pca[:50, None] - pca[None, :50], axis=-1))
    st = Static("synth", n, "A" * n, ["A1G"] * n, emb, nbrs, pca, {"zs_good": fitness + 0.01 * rng.standard_normal(n)})
    return st, fitness


def test_oracle_counts_unique_queries_and_raises_when_spent():
    o = Oracle(np.arange(10.0), budget=3)
    assert [o.query(i) for i in (1, 1, 2, 2, 1)] == [1.0, 1.0, 2.0, 2.0, 1.0] and o.n_queries == 2
    o.query(5)
    with pytest.raises(BudgetExhausted):
        o.query(6)
    assert o.query(5) == 5.0 and o.log == [1, 2, 5]  # already-owned measurements stay readable


def test_static_has_no_fitness_so_methods_cannot_read_it():
    st, _ = synth()
    assert not any("fit" in k.lower() for k in vars(st))


def cfgs(st):
    return {**{m: {} for m in METHODS}, "fly": {"codes": st.codes(0, n_kc=200)}, "fly_mem": {"codes": st.codes(0, n_kc=200), "memory": True},
            "zeroshot": {"scores_key": "zs_good"}}


def fn_of(name):
    return {"fly": fly, "fly_mem": fly, "zeroshot": zeroshot}.get(name) or METHODS[name]


def test_all_methods_respect_budget_are_deterministic_and_prefix_consistent():
    st, f = synth()
    for name, cfg in cfgs(st).items():
        a = run_method(fn_of(name), st, f, 60, 7, np.random.default_rng(3), **cfg)
        b = run_method(fn_of(name), st, f, 60, 7, np.random.default_rng(3), **cfg)
        short = run_method(fn_of(name), st, f, 25, 7, np.random.default_rng(3), **cfg)
        assert a == b, name  # deterministic logging
        assert len(a) == len(set(a)) <= 60, name  # unique queries within budget
        assert a[: len(short)] == short and len(short) <= 25, name  # a smaller budget is a prefix: nothing depends on the budget


def test_zeroshot_queries_in_score_order_without_feedback():
    st, f = synth()
    log = run_method(zeroshot, st, f, 10, 0, np.random.default_rng(0), scores_key="zs_good")
    assert log == list(np.argsort(-st.zs["zs_good"], kind="stable")[:10])


def test_gp_bo_beats_random_search_on_a_smooth_landscape():
    st, f = synth()
    gp, rs = [], []
    for s in range(6):
        for name, acc in (("gpbo_ei", gp), ("random_search", rs)):
            log = run_method(METHODS[name], st, f, 40, int(np.random.default_rng(s).integers(st.n)), np.random.default_rng(s))
            acc.append(curve_percentiles(log, f, 40)[-1])
    assert np.mean(gp) > np.mean(rs)


def test_greedy_local_climbs_to_the_ridge_from_a_nearby_start():
    st, f = synth()
    peak = int(np.argmax(f))
    log = run_method(METHODS["greedy_local"], st, f, 100, peak - 25, np.random.default_rng(0))
    assert f[log].max() == f.max()  # steepest ascent over +-7 neighbours reaches the peak within ~4 steps


def test_curve_percentiles():
    f = np.arange(100.0)
    c = curve_percentiles([3, 50, 10, 99], f, 6)
    assert c[0] == 4.0 and c[1] == 51.0 and c[2] == 51.0 and c[3] == 100.0 and np.isnan(c[4])


def test_tracing_does_not_change_the_run_and_records_consistent_state():
    from flyprotein.bench.gpbo import gpbo

    st, f = synth()
    codes = st.codes(0, n_kc=200)
    t_fly, t_bo = [], []
    a = run_method(fly, st, f, 60, 7, np.random.default_rng(3), codes=codes)
    b = run_method(fly, st, f, 60, 7, np.random.default_rng(3), codes=codes, trace=t_fly)
    assert a == b
    c = run_method(gpbo, st, f, 40, 7, np.random.default_rng(3))
    d = run_method(gpbo, st, f, 40, 7, np.random.default_rng(3), trace=t_bo)
    assert c == d
    # recorded events reproduce the query log, and replaying the recorded weight increments reproduces the scores
    assert [e["pos"] for e in t_fly if e["type"] in ("init", "restart")][0] == 7
    w = np.zeros(200)
    for e in t_fly:
        if e["type"] == "move":
            assert np.allclose(w[codes[st.nbrs[e["pos"]]]].sum(1), e["scores"])
            w[codes[e["nxt"]]] += e["a"]
            w[codes[e["pos"]]] -= e["a"]
    assert [e["chosen"] for e in t_bo] == d
