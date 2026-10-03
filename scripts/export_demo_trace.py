"""Record the live-demo trajectories (TEM-1, seed 0, 200 queries, same start for both methods) from the real implementations.

    python scripts/export_demo_trace.py
Writes web/public/data/demo_landscape.json (map + fitness) and web/public/data/demo_trace.json (recorded search state).
The traced runs are verified to equal the benchmark's query logs (results/benchmark/queries.npz) before anything is written, so
what the page replays is exactly the benchmark run; the page only replays this recorded state.
"""
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_benchmark import rng_for  # noqa: E402

from flyprotein.bench import load_assay_data, make_static  # noqa: E402
from flyprotein.bench.data import MODEL  # noqa: E402
from flyprotein.bench.gpbo import gpbo  # noqa: E402
from flyprotein.bench.methods import fly, run_method  # noqa: E402
from flyprotein.bench.metrics import percentile_table  # noqa: E402

ASSAY, SEED, BUDGET = "BLAT_ECOLX_Firnberg_2014", 0, 200  # fixed a priori: the project's original landscape, seed 0 of 20
OUT = ROOT / "web/public/data"

st = make_static(ASSAY)
f = load_assay_data(ASSAY)
start = int(np.random.default_rng(1000 + SEED).integers(st.n))
codes = st.codes(SEED)
bench = np.load(ROOT / "results/benchmark/queries.npz")

fly_trace, bo_trace = [], []
fly_log = run_method(fly, st, f, BUDGET, start, rng_for(SEED, "fly"), codes=codes, trace=fly_trace)
bo_log = run_method(gpbo, st, f, BUDGET, start, rng_for(SEED, "gpbo_ei"), trace=bo_trace)
for name, log in (("fly", fly_log), ("gpbo_ei", bo_log)):
    ref = bench[f"{ASSAY}|{name}|{SEED}"][:BUDGET].tolist()
    assert log == ref, f"{name}: traced run differs from the benchmark query log"
print("traced runs identical to benchmark logs:", len(fly_log), len(bo_log), "queries")

# landscape for the map: UMAP of the 8M embeddings (display only; the search walks the 15-NN graph, not this 2-D picture)
cache = ROOT / f"data/cache/{ASSAY}.umap.npy"
if cache.exists():
    xy = np.load(cache)
else:
    import umap

    xy = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, metric="cosine", random_state=0).fit_transform(st.emb - st.emb.mean(0))
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    np.save(cache, xy)
pct = percentile_table(f)(f)
landscape = dict(assay=ASSAY, n=st.n, mutants=st.mutants,
                 x=[round(float(v), 4) for v in xy[:, 0]], y=[round(float(v), 4) for v in xy[:, 1]],
                 fitness=[round(float(v), 3) for v in f], pct=[round(float(v), 2) for v in pct])
(OUT / "demo_landscape.json").write_text(json.dumps(landscape, separators=(",", ":")))

visited = sorted({e["pos"] for e in fly_trace} | {e["nxt"] for e in fly_trace if e["type"] == "move"})
Ec = st.emb - st.emb.mean(0)
scale = float(np.abs(Ec[visited]).max())
trace = dict(
    assay=ASSAY, seed=SEED, budget=BUDGET, start=start, n_kc=int(codes.max()) + 1, ka=int(codes.shape[1]), embedding=MODEL, pn_scale=scale,
    fly=dict(events=fly_trace,
             codes={str(i): codes[i].tolist() for i in visited},
             nbrs={str(i): st.nbrs[i].tolist() for i in sorted({e["pos"] for e in fly_trace if e["type"] == "move"})},
             pn={str(i): np.round(Ec[i] / np.abs(Ec[i]).max() * 127).astype(int).tolist() for i in visited}),  # each variant scaled to its own largest channel (display only)
    gpbo=dict(events=bo_trace),
    verified=dict(fly_log_sha256=hashlib.sha256(np.array(fly_log, dtype=np.int32).tobytes()).hexdigest(),
                  gpbo_log_sha256=hashlib.sha256(np.array(bo_log, dtype=np.int32).tobytes()).hexdigest(),
                  note="identical to results/benchmark/queries.npz entries for this assay and seed"),
)
(OUT / "demo_trace.json").write_text(json.dumps(trace, separators=(",", ":")))
print("fly events:", len(fly_trace), "moves:", sum(e["type"] == "move" for e in fly_trace), "revisit moves:", sum(e["type"] == "move" and not e["new"] for e in fly_trace))
print("fly best pct@50/100/200:", [round(float(percentile_table(f)(f[fly_log[:b]].max())), 2) for b in (50, 100, 200)])
print("gpbo best pct@50/100/200:", [round(float(percentile_table(f)(f[bo_log[:b]].max())), 2) for b in (50, 100, 200)])
for p in sorted(OUT.glob("demo_*.json")):
    print(p.name, f"{p.stat().st_size / 1e3:.0f} KB")
