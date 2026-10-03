"""Run every method on every benchmark assay with shared seeds; save the deterministic query logs.

    python scripts/run_benchmark.py --seeds 20 --budget 500

Per (assay, seed): one shared start variant (default_rng(1000+seed)); every method gets the same start and a method-specific
RNG stream derived from (seed, method). Outputs results/benchmark/queries.npz (one int array per run, key 'assay|method|seed').
Methods never see fitness: they only receive an Oracle (src/flyprotein/bench/oracle.py).
"""
import argparse
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from embed_assays import ASSAYS  # noqa: E402

from flyprotein.bench import load_assay_data, make_static  # noqa: E402
from flyprotein.bench.methods import METHODS, fly, run_method, zeroshot  # noqa: E402

# the best fly configuration from the (already finished) learner sweep: replay 5 + temperature 0.3 + memory
FLY = {
    "fly": dict(),
    "fly_memory": dict(memory=True),
    "fly_best_tested": dict(memory=True, replay=5, T=0.3),
}


def rng_for(seed, name):
    return np.random.default_rng((seed, zlib.crc32(name.encode())))


def job(args):
    assay, seeds, budget = args
    t0 = time.time()
    st = make_static(assay)
    f = load_assay_data(assay)
    out = {}
    zs_methods = {f"zs_{k.split('.')[1]}_{k.split('_')[2]}": k for k in st.zs}  # e.g. zs_wt_150M
    for seed in range(seeds):
        start = int(np.random.default_rng(1000 + seed).integers(st.n))
        for name, fn in METHODS.items():
            out[f"{assay}|{name}|{seed}"] = np.array(run_method(fn, st, f, budget, start, rng_for(seed, name)), dtype=np.int32)
        codes = st.codes(seed)
        for name, cfg in FLY.items():
            out[f"{assay}|{name}|{seed}"] = np.array(run_method(fly, st, f, budget, start, rng_for(seed, name), codes=codes, **cfg), dtype=np.int32)
        for name, key in zs_methods.items():
            out[f"{assay}|{name}|{seed}"] = np.array(run_method(zeroshot, st, f, budget, start, rng_for(seed, name), scores_key=key), dtype=np.int32)
    print(f"{assay} done in {time.time() - t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--assays", nargs="+", default=ASSAYS)
    ap.add_argument("--out", default=str(ROOT / "results/benchmark/queries.npz"))
    a = ap.parse_args()
    with Pool(a.procs) as p:
        runs = {}
        for part in p.imap_unordered(job, [(x, a.seeds, a.budget) for x in a.assays]):
            runs.update(part)
    np.savez_compressed(a.out, **runs)
    print("wrote", a.out, len(runs), "runs")
