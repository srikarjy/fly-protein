"""Serialize the circuit state that the brain/circuit view of the demo needs, from the real implementation.

    python scripts/export_demo_circuit.py
Writes web/public/data/demo_circuit.json. This is purely additive display serialization: it does NOT run or change any search,
and it leaves web/public/data/demo_trace.json and demo_landscape.json (and results/) byte-identical. It adds what the trace
does not carry:

  wiring   the PN->KC projection the benchmark used for this seed (random-sparse FlyHash, 32 of 320 input channels per KC)
  drive    the exact input drive (sum of the wired PN values) of each active Kenyon cell of every measured variant,
           plus the winner-take-all cut-off (weakest active / strongest inactive drive)
  pn_max   scale that turns the trace's 8-bit PN values back into embedding units
  probs    the move probabilities the learner's softmax rule assigned to the 15 neighbours at every recorded move

Everything is recomputed from the same Static/FlyHash code the benchmark uses and asserted to reproduce the committed trace
(codes of every measured variant, the neighbour lists, the recorded 8-bit PN values) before anything is written.
"""
import base64
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from flyprotein.bench import make_static  # noqa: E402
from flyprotein.bench.methods import fly  # noqa: E402
from flyprotein.flyhash import FlyHash  # noqa: E402

OUT = ROOT / "web/public/data"
TRACE_PATH = OUT / "demo_trace.json"

trace_bytes = TRACE_PATH.read_bytes()
trace = json.loads(trace_bytes)
ASSAY, SEED = trace["assay"], trace["seed"]
n_kc, ka = trace["n_kc"], trace["ka"]
T = float(inspect.signature(fly).parameters["T"].default)  # the temperature of the recorded run

st = make_static(ASSAY)
Ec = st.emb - st.emb.mean(0)  # exactly what Static.codes() feeds to FlyHash
d_in = Ec.shape[1]
fh = FlyHash(d_in=d_in, n_kc=n_kc, active_frac=ka / n_kc, seed=SEED)

# 1. the wiring must reproduce the committed codes
codes_all = st.codes(SEED)
assert (np.nonzero(fh.encode(Ec))[1].reshape(st.n, ka) == codes_all).all(), "wiring does not reproduce the benchmark codes"
per_kc = int(fh.W[0].sum())
assert (fh.W.sum(1) == per_kc).all() and set(np.unique(fh.W)) <= {0.0, 1.0}
wiring = np.nonzero(fh.W)[1].reshape(n_kc, per_kc).astype("<u2")  # row = KC, ascending PN index

visited = sorted(int(k) for k in trace["fly"]["codes"])
for i in visited:
    assert trace["fly"]["codes"][str(i)] == codes_all[i].tolist(), f"trace codes differ for variant {i}"

# 2. exact drive of the active cells, and the cut-off that separates winners from the rest
act = Ec[visited] @ fh.W.T  # (n_visited, n_kc), the quantity FlyHash.encode ranks
drive, cutoff, pn_max = {}, {}, {}
for r, i in enumerate(visited):
    idx = np.array(trace["fly"]["codes"][str(i)])
    on = np.zeros(n_kc, bool)
    on[idx] = True
    assert act[r][on].min() >= act[r][~on].max(), "active cells are not the top-drive cells"
    drive[str(i)] = [round(float(v), 5) for v in act[r][idx]]
    cutoff[str(i)] = [round(float(act[r][on].min()), 5), round(float(act[r][~on].max()), 5)]
    pn_max[str(i)] = round(float(np.abs(Ec[i]).max()), 6)
    q = np.round(Ec[i] / np.abs(Ec[i]).max() * 127).astype(int).tolist()
    assert q == trace["fly"]["pn"][str(i)], f"8-bit PN values differ for variant {i}"

# 3. the learner's move probabilities (the rule the Python run sampled from), per recorded event
probs = []
for e in trace["fly"]["events"]:
    if e["type"] != "move":
        probs.append(None)
        continue
    s = np.array(e["scores"])
    p = np.exp((s - s.max()) / T)
    probs.append([round(float(v), 6) for v in p / p.sum()])
    assert st.nbrs[e["pos"]].tolist() == trace["fly"]["nbrs"][str(e["pos"])]

circuit = dict(
    assay=ASSAY, seed=SEED,
    trace_sha256=hashlib.sha256(trace_bytes).hexdigest(),
    meta=dict(n_pn=int(d_in), n_kc=n_kc, per_kc=per_kc, k_active=ka, temperature=T,
              wiring="random-sparse FlyHash wiring (each Kenyon cell samples %d of %d input channels, weight 1), seed %d; the real hemibrain wiring is not used in this run" % (per_kc, d_in, SEED),
              pn="the 320 mean-centred ESM-2 8M embedding channels of the variant (the FlyHash input); 8-bit values in the trace times pn_max/127"),
    wiring_b64=base64.b64encode(wiring.tobytes()).decode(),
    pn_max=pn_max, drive=drive, cutoff=cutoff, probs=probs,
)
(OUT / "demo_circuit.json").write_text(json.dumps(circuit, separators=(",", ":")))
size = (OUT / "demo_circuit.json").stat().st_size
print(f"wrote demo_circuit.json: {size / 1e3:.0f} KB; wiring {wiring.shape}, {len(visited)} variants, {sum(p is not None for p in probs)} move probability vectors, T={T}")
