"""Deterministic replay in CI: every search method, run on a fixed synthetic landscape, must reproduce the committed
expected query trajectories exactly. Regenerate (deliberately) with:  UPDATE_GOLDEN=1 pytest tests/test_replay_golden.py
The full-data counterpart is scripts/verify_replay.py (needs the cached embeddings)."""
import json
import os
from pathlib import Path

import numpy as np

from flyprotein.bench.methods import METHODS, fly, run_method, zeroshot
from test_bench import synth

GOLDEN = Path(__file__).parent / "golden/replay_synthetic.json"
BUDGET = 40


def replay():
    st, f = synth()
    codes = st.codes(0, n_kc=200)
    start = 11
    jobs = {name: (fn, {}) for name, fn in METHODS.items()}
    jobs["fly"] = (fly, dict(codes=codes))
    jobs["fly_memory"] = (fly, dict(codes=codes, memory=True))
    jobs["fly_best_tested"] = (fly, dict(codes=codes, memory=True, replay=5, T=0.3))
    jobs["zeroshot"] = (zeroshot, dict(scores_key="zs_good"))
    return {name: [int(i) for i in run_method(fn, st, f, BUDGET, start, np.random.default_rng((3, len(name))), **cfg)] for name, (fn, cfg) in jobs.items()}


def test_every_method_reproduces_its_committed_trajectory():
    got = replay()
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(got, indent=0))
    want = json.loads(GOLDEN.read_text())
    assert set(got) == set(want)
    for name in want:
        assert got[name] == want[name], f"{name}: trajectory changed (nondeterminism or an intentional method change)"
        assert len(got[name]) == len(set(got[name])) == BUDGET
