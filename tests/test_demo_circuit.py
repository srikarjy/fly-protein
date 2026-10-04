"""The circuit view's serialized state (web/public/data/demo_circuit.json) must be internally consistent and bound to the
committed trace it was exported against. Needs no embedding cache: it checks the committed JSON files against each other."""
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parent.parent / "web/public/data"
trace_bytes = (DATA / "demo_trace.json").read_bytes()
trace = json.loads(trace_bytes)
circ = json.loads((DATA / "demo_circuit.json").read_text())
M = circ["meta"]
wiring = np.frombuffer(base64.b64decode(circ["wiring_b64"]), dtype="<u2").reshape(M["n_kc"], M["per_kc"])


def test_bound_to_the_committed_trace():
    assert circ["trace_sha256"] == hashlib.sha256(trace_bytes).hexdigest()
    assert (circ["assay"], circ["seed"], M["n_kc"], M["k_active"]) == (trace["assay"], trace["seed"], trace["n_kc"], trace["ka"])
    assert set(circ["drive"]) == set(trace["fly"]["codes"]) == set(circ["pn_max"]) == set(circ["cutoff"])


def test_wiring_is_a_valid_sparse_projection():
    assert wiring.shape == (2000, 32)
    assert wiring.max() < M["n_pn"] == 320
    assert all(len(set(row)) == M["per_kc"] for row in wiring)  # each KC samples distinct input channels
    assert (np.diff(wiring.astype(int), axis=1) > 0).all()  # ascending, as exported


def test_active_cell_drive_equals_the_sum_of_its_wired_inputs():
    """drive of an active KC = sum of the PN values it samples. The trace stores 8-bit PN values, so allow the quantisation bound."""
    for key, drive in circ["drive"].items():
        codes = trace["fly"]["codes"][key]
        pn = np.array(trace["fly"]["pn"][key]) / 127 * circ["pn_max"][key]
        tol = M["per_kc"] * 0.5 / 127 * circ["pn_max"][key] + 1e-4
        assert len(drive) == len(codes) == M["k_active"]
        for kc, d in zip(codes, drive):
            assert abs(pn[wiring[kc]].sum() - d) <= tol
        lo, hi = circ["cutoff"][key]
        assert min(drive) == pytest.approx(lo, abs=1e-5) and lo >= hi  # winners all drive at least as hard as every loser


def test_recorded_move_probabilities_are_the_softmax_of_the_recorded_scores():
    T = M["temperature"]
    events = trace["fly"]["events"]
    assert len(circ["probs"]) == len(events)
    for e, p in zip(events, circ["probs"]):
        if e["type"] != "move":
            assert p is None
            continue
        s = np.array(e["scores"])
        want = np.exp((s - s.max()) / T)
        assert np.allclose(p, want / want.sum(), atol=2e-6)
        assert sum(p) == pytest.approx(1, abs=2e-5)
