"""Deterministic replay: re-run benchmark jobs and compare every query log with results/benchmark/queries.npz.

    python scripts/verify_replay.py                      # one assay, 2 seeds (default)
    python scripts/verify_replay.py --assay BLAT_ECOLX_Firnberg_2014 --seeds 20
Exit status 1 if any log differs. Needs the cached embeddings/zero-shot scores in data/cache (see README).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_benchmark as rb  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", default="UBC9_HUMAN_Weile_2017")
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    ref = np.load(ROOT / "results/benchmark/queries.npz")
    out = rb.job((a.assay, a.seeds, 500))
    bad = [k for k, v in out.items() if not np.array_equal(v, ref[k])]
    print(f"{len(out)} runs replayed, {len(bad)} differ" + (": " + ", ".join(bad[:5]) if bad else " -> deterministic replay OK"))
    sys.exit(1 if bad else 0)
