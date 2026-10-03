"""Zero-shot ESM-2 scores for single substitutions. Never touches measured fitness.

WT-marginal  (Meier et al. 2021):  s(i, a->b) = log p(x_i = b | x_WT) - log p(x_i = a | x_WT)
             one forward pass of the unmasked wild-type sequence per assay.
Masked-marginal:                   s(i, a->b) = log p(x_i = b | x_WT with position i masked) - log p(x_i = a | same)
             one forward pass per position.
Positions are 1-indexed as in ProteinGym mutation strings; token index = position because of the leading <cls>.

    python scripts/zeroshot_scores.py --model facebook/esm2_t33_650M_UR50D --mode wt mm
Writes data/cache/<assay>.zs.<model>.<mode>.npy (aligned with <assay>.variants.json). The Spearman against
measured fitness is computed afterwards ONLY as a sanity check of the implementation (results/benchmark/zeroshot_sanity.csv).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from embed_assays import ASSAYS  # noqa: E402


def score(model, tok, wt, mutants, mode, batch=8):
    ids = tok(wt, return_tensors="pt")["input_ids"]  # (1, L+2)
    aa = lambda c: tok.convert_tokens_to_ids(c)
    pos_list = sorted({int(m[1:-1]) for m in mutants})
    lp = {}  # position -> log-probs over vocab
    with torch.no_grad():
        if mode == "wt":
            logp = torch.log_softmax(model(input_ids=ids).logits[0], -1)
            lp = {p: logp[p] for p in pos_list}
        else:
            for i in range(0, len(pos_list), batch):
                chunk = pos_list[i : i + batch]
                x = ids.repeat(len(chunk), 1)
                for r, p in enumerate(chunk):
                    x[r, p] = tok.mask_token_id
                logp = torch.log_softmax(model(input_ids=x).logits, -1)
                for r, p in enumerate(chunk):
                    lp[p] = logp[r, p]
    return np.array([float(lp[int(m[1:-1])][aa(m[-1])] - lp[int(m[1:-1])][aa(m[0])]) for m in mutants])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--mode", nargs="+", default=["wt"])
    ap.add_argument("--assays", nargs="+", default=ASSAYS)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = EsmForMaskedLM.from_pretrained(a.model).eval()
    short = a.model.split("/")[-1]
    for assay in a.assays:
        meta = json.load(open(ROOT / f"data/cache/{assay}.variants.json"))
        for mode in a.mode:
            t = time.time()
            s = score(model, tok, meta["wt"], meta["mutants"], mode)
            np.save(ROOT / f"data/cache/{assay}.zs.{short}.{mode}.npy", s)
            print(assay, short, mode, f"{time.time() - t:.0f}s", flush=True)
