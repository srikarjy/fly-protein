"""Cache variant lists + wild-type sequences for the benchmark assays (rows match the embedding/fitness caches)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "prep"))
sys.path.insert(0, str(ROOT / "scripts"))
from build_data import load_assay  # noqa: E402
from embed_assays import ASSAYS  # noqa: E402

for a in ASSAYS:
    df = load_assay(a, str(ROOT / "data/proteingym/DMS_substitutions/*.parquet"))
    wt = df.target_seq.iloc[0]
    muts = df.mutant.tolist()
    assert not any(":" in m for m in muts), "benchmark assumes single substitutions"
    assert all(wt[int(m[1:-1]) - 1] == m[0] for m in muts), f"{a}: WT residue mismatch"
    (ROOT / f"data/cache/{a}.variants.json").write_text(json.dumps({"mutants": muts, "wt": wt}))
    print(a, len(muts), "WT length", len(wt))
