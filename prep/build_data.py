"""Build web/public/data.json: one ProteinGym DMS assay embedded, mapped and fingerprinted.

    python prep/build_data.py --assay BLAT_ECOLX_Firnberg_2014 --device cpu

Steps: load assay -> ESM-2 embed -> UMAP (fixed seed) -> kNN in the ORIGINAL
embedding space -> FlyHash fingerprints -> data.json. The fly walks the kNN
graph; the 2D map is only how the graph is drawn.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from flyprotein.flyhash import FlyHash
from flyprotein.project import gaussian_projection

ROOT = Path(__file__).resolve().parent.parent


def emb_cache_path(cache: Path, assay: str, model: str) -> Path:
    """Embeddings are cached per model so switching checkpoints can't reuse stale vectors."""
    return cache / f"{assay}.{model.split('/')[-1]}.emb.npy"


def load_assay(assay: str, shard_glob: str) -> "pd.DataFrame":
    import pandas as pd

    parts = []
    for f in sorted(glob.glob(shard_glob)):
        df = pd.read_parquet(f, columns=["DMS_id", "mutant", "mutated_sequence", "target_seq", "DMS_score"])
        df = df[df.DMS_id == assay]
        if len(df):
            parts.append(df)
    if not parts:
        raise SystemExit(f"assay {assay!r} not found under {shard_glob}")
    df = pd.concat(parts).drop_duplicates("mutant").reset_index(drop=True)
    return df.dropna(subset=["DMS_score"]).reset_index(drop=True)


def knn_cosine(X: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest rows by cosine similarity (self excluded)."""
    Z = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-9)
    out = np.empty((len(Z), k), dtype=np.int32)
    for i in range(0, len(Z), 1024):
        sim = Z[i : i + 1024] @ Z.T
        sim[np.arange(sim.shape[0]), np.arange(i, i + sim.shape[0])] = -np.inf
        idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sim, idx, axis=1), axis=1)
        out[i : i + 1024] = np.take_along_axis(idx, order, axis=1)
    return out


def hemibrain_codes(E: np.ndarray, edges, n_pn: int, n_kc: int, seed: int = 0) -> tuple[np.ndarray, int]:
    """Sparse codes through real PN->KC wiring. E (n, d) -> project to n_pn PN channels
    (fixed seeded Gaussian projection) -> synapse-count matrix -> top 5% of KCs."""
    P = gaussian_projection(E.shape[1], n_pn, seed=seed)
    Ep = E @ P
    fh = FlyHash.from_edges(edges, n_pn, n_kc).fit(Ep)
    return fh.encode(Ep), fh.k_active


def add_hemibrain(path: Path, cache: Path, seed: int = 0) -> None:
    """Add real-wiring codes to an existing data.json without touching positions or neighbors."""
    from flyprotein.connectome import load_pn_kc_edges

    raw = json.loads(path.read_text())
    assay = raw["meta"]["assay"]
    E = np.load(emb_cache_path(cache, assay, raw["meta"]["embedding"]))
    if E.shape[0] != len(raw["variants"]):
        raise SystemExit("cached embeddings do not match data.json; rebuild without --hemi-only")
    edges, n_pn, n_kc = load_pn_kc_edges(cache=str(cache / "hemibrain_pn_kc.npz"))
    codes, k_active = hemibrain_codes(E, edges, n_pn, n_kc, seed)
    for v, c in zip(raw["variants"], codes):
        v["h"] = np.flatnonzero(c).tolist()
    raw["meta"]["hemi"] = {
        "n_pn": n_pn,
        "n_kc": n_kc,
        "active_kc": k_active,
        "n_edges": len(edges),
        "proj_seed": seed,
        "dataset": "hemibrain:v1.2.1",
        "roi": "CA(R)",
        "note": "Real PN->KC synapse counts (type .*PN.* onto KC.*, includes a few non-olfactory PNs). "
        "Input is a fixed random Gaussian projection of the ESM-2 embedding to the PN count.",
    }
    path.write_text(json.dumps(raw, separators=(",", ":")))
    print(f"hemibrain: {n_pn} PNs -> {n_kc} KCs, {k_active} active; wrote {path} ({path.stat().st_size/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", default="BLAT_ECOLX_Firnberg_2014")
    ap.add_argument("--shards", default=str(ROOT / "data/proteingym/DMS_substitutions/*.parquet"))
    ap.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    ap.add_argument("--device", default=None)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--n-kc", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "web/public/data.json"))
    ap.add_argument("--cache", default=str(ROOT / "data/cache"))
    ap.add_argument("--hemi-only", action="store_true", help="add real-wiring codes to existing data.json and exit")
    a = ap.parse_args()

    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    if a.hemi_only:
        add_hemibrain(Path(a.out), cache, a.seed)
        return
    df = load_assay(a.assay, a.shards)
    print(f"{a.assay}: {len(df)} variants, DMS_score {df.DMS_score.min():.3f}..{df.DMS_score.max():.3f}")
    wt = df.target_seq.iloc[0]

    emb_path = emb_cache_path(cache, a.assay, a.model)
    if emb_path.exists() and np.load(emb_path).shape[0] == len(df):
        E = np.load(emb_path)
    else:
        from flyprotein.embed import ESM2Embedder

        emb = ESM2Embedder(a.model, device=a.device, batch_size=16)
        E = emb(df.mutated_sequence.tolist()).astype(np.float32)
        np.save(emb_path, E)
    print("embeddings", E.shape)

    Ec = E - E.mean(axis=0)  # same centering FlyHash applies; drop the shared component
    nbrs = knn_cosine(Ec, a.k)

    import umap

    xy = umap.UMAP(n_components=2, n_neighbors=a.k, min_dist=0.3, metric="cosine", random_state=a.seed).fit_transform(Ec)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))

    fh = FlyHash(d_in=E.shape[1], n_kc=a.n_kc, seed=a.seed).fit(E)
    codes = fh.encode(E)

    variants = [
        {
            "id": i,
            "m": df.mutant.iloc[i],
            "f": round(float(df.DMS_score.iloc[i]), 4),
            "x": round(float(xy[i, 0]), 4),
            "y": round(float(xy[i, 1]), 4),
            "n": nbrs[i].tolist(),
            "c": np.flatnonzero(codes[i]).tolist(),
        }
        for i in range(len(df))
    ]
    out = {
        "meta": {
            "assay": a.assay,
            "source": "OATML-Markslab/ProteinGym_v1 (DMS_substitutions)",
            "embedding": a.model,
            "wt": wt,
            "k": a.k,
            "n_kc": a.n_kc,
            "active_kc": fh.k_active,
            "umap_seed": a.seed,
            "note": "The 2D map is a lossy UMAP picture. Neighbors are the real kNN graph in the original embedding space.",
        },
        "variants": variants,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {a.out} ({Path(a.out).stat().st_size/1e6:.1f} MB)")

    # M1 check: static scatter coloured by fitness
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=df.DMS_score, s=4, cmap="viridis")
        fig.colorbar(sc, label="DMS_score")
        ax.set_title(a.assay)
        fig.savefig(cache / f"{a.assay}.map.png", dpi=120, bbox_inches="tight")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
