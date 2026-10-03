"""Embed several ProteinGym assays with ESM-2 8M on CPU and cache them (rows match prep/build_data.load_assay).

    python scripts/embed_assays.py
Writes data/cache/<assay>.esm2_t6_8M_UR50D.emb.npy and data/cache/<assay>.fitness.npy.
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "prep"))
from build_data import emb_cache_path, load_assay  # noqa: E402

from flyprotein.embed import ESM2Embedder  # noqa: E402

ASSAYS = [
    "BLAT_ECOLX_Firnberg_2014", "DYR_ECOLI_Nguyen_2023", "RASH_HUMAN_Bandaru_2017", "UBC9_HUMAN_Weile_2017",
    "NUD15_HUMAN_Suiter_2020", "TPK1_HUMAN_Weile_2017", "MLAC_ECOLI_MacRae_2023", "KKA2_KLEPN_Melnikov_2014",
    "TPMT_HUMAN_Matreyek_2018", "RNC_ECOLI_Weeks_2023",
]
MODEL = "facebook/esm2_t6_8M_UR50D"

if __name__ == "__main__":
    cache = ROOT / "data/cache"
    emb = None
    for a in ASSAYS:
        df = load_assay(a, str(ROOT / "data/proteingym/DMS_substitutions/*.parquet"))
        np.save(cache / f"{a}.fitness.npy", df.DMS_score.to_numpy(dtype=np.float64))
        p = emb_cache_path(cache, a, MODEL)
        if p.exists() and np.load(p).shape[0] == len(df):
            print(a, "cached", len(df), flush=True)
            continue
        emb = emb or ESM2Embedder(MODEL, device="cpu", batch_size=32)
        t = time.time()
        E = emb(df.mutated_sequence.tolist()).astype(np.float32)
        np.save(p, E)
        print(a, len(df), E.shape, f"{time.time() - t:.0f}s", flush=True)
