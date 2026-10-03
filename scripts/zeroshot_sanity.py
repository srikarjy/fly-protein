"""Spearman between zero-shot scores and measured fitness: an implementation sanity check, never used by any method."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from embed_assays import ASSAYS  # noqa: E402

rows = []
for a in ASSAYS:
    f = np.load(ROOT / f"data/cache/{a}.fitness.npy")
    for p in sorted((ROOT / "data/cache").glob(f"{a}.zs.*.npy")):
        model, mode = p.name.split(".zs.")[1][:-4].rsplit(".", 1)
        rows.append(dict(assay=a.split("_")[0], model=model.split("_")[-2], mode=mode, spearman=spearmanr(np.load(p), f).statistic))
d = pd.DataFrame(rows)
t = d.pivot_table(index="assay", columns=["mode", "model"], values="spearman").round(3)
t.loc["mean"] = t.mean().round(3)
print(t.to_string())
d.to_csv(ROOT / "results/benchmark/zeroshot_sanity.csv", index=False)
