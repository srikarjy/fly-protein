"""Fetch real PN->KC wiring from the hemibrain via neuPrint.

NOT yet run against the live server. It follows the neuprint-python API
(Client, fetch_adjacencies, NeuronCriteria). Needs a neuPrint auth token in
the NEUPRINT_APPLICATION_CREDENTIALS env var, and network access to
neuprint.janelia.org.
"""
from __future__ import annotations

import numpy as np


def fetch_pn_kc_edges(dataset: str = "hemibrain:v1.2.1", server: str = "neuprint.janelia.org"):
    """Return (edges, n_pn, n_kc) where edges are (pn_idx, kc_idx, weight)."""
    from neuprint import Client, NeuronCriteria as NC, fetch_adjacencies

    Client(server, dataset=dataset)
    neurons, conn = fetch_adjacencies(
        sources=NC(type=".*PN.*", regex=True),
        targets=NC(type="KC.*", regex=True),
        rois=["CA(R)"],  # mushroom body calyx, where PNs synapse onto KCs
        min_total_weight=1,
    )
    conn = conn.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
    pns = sorted(conn.bodyId_pre.unique())
    kcs = sorted(conn.bodyId_post.unique())
    pn_idx = {b: i for i, b in enumerate(pns)}
    kc_idx = {b: i for i, b in enumerate(kcs)}
    edges = [(pn_idx[r.bodyId_pre], kc_idx[r.bodyId_post], r.weight) for r in conn.itertuples()]
    return edges, len(pns), len(kcs)


def save_edges(path: str, edges, n_pn: int, n_kc: int) -> None:
    np.savez(path, edges=np.array(edges, dtype=np.float64), n_pn=n_pn, n_kc=n_kc)


def load_edges(path: str):
    z = np.load(path)
    return [tuple(e) for e in z["edges"]], int(z["n_pn"]), int(z["n_kc"])
