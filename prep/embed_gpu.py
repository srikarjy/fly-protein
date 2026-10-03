# /// script
# dependencies = ["torch", "transformers", "pandas", "pyarrow", "huggingface_hub", "numpy"]
# ///
"""Embed one ProteinGym assay with ESM-2 on a GPU (Colab/Kaggle T4 or any CUDA box).

Self-contained: needs only torch, transformers, pandas, pyarrow, huggingface_hub.

    python embed_gpu.py --assay BLAT_ECOLX_Firnberg_2014 --model facebook/esm2_t33_650M_UR50D

Writes <assay>.<model>.emb.npy, the exact filename prep/build_data.py looks for in
data/cache/. Row order matches build_data.load_assay (sorted shards, filter assay,
drop duplicate mutants, drop missing scores), mean-pooled over residues (CLS/EOS excluded)
like flyprotein.embed.ESM2Embedder, so results are comparable with the CPU runs.
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

REPO = "OATML-Markslab/ProteinGym_v1"
SHARDS = [f"DMS_substitutions/train-{i:05d}-of-00005.parquet" for i in range(5)]


def load_assay(assay):
    parts = []
    for s in SHARDS:
        df = pd.read_parquet(hf_hub_download(REPO, s, repo_type="dataset"), columns=["DMS_id", "mutant", "mutated_sequence", "DMS_score"])
        df = df[df.DMS_id == assay]
        if len(df):
            parts.append(df)
    if not parts:
        raise SystemExit(f"assay {assay!r} not found")
    df = pd.concat(parts).drop_duplicates("mutant").reset_index(drop=True)
    return df.dropna(subset=["DMS_score"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", default="BLAT_ECOLX_Firnberg_2014")
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--push-to", help="private HF dataset repo id to upload the .npy to (needs HF_TOKEN); for HF Jobs, which have no persistent disk")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "(no GPU: this will be slow)")
    df = load_assay(a.assay)
    seqs = df.mutated_sequence.tolist()
    print(f"{a.assay}: {len(seqs)} variants")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    out, t0 = [], time.time()
    for i in range(0, len(seqs), a.batch):
        enc = tok(seqs[i : i + a.batch], return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            h = model(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].clone()
        lengths = mask.sum(1)
        mask[:, 0] = 0
        mask[torch.arange(mask.shape[0]), lengths - 1] = 0
        m = mask.unsqueeze(-1).to(h.dtype)
        out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
        if (i // a.batch) % 10 == 0:
            print(f"{i + len(out[-1])}/{len(seqs)}  {time.time() - t0:.0f}s", flush=True)
    E = np.concatenate(out).astype(np.float32)
    path = f"{a.assay}.{a.model.split('/')[-1]}.emb.npy"
    np.save(path, E)
    print("saved", path, E.shape, f"in {time.time() - t0:.0f}s")
    if a.push_to:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(a.push_to, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=path, path_in_repo=path, repo_id=a.push_to, repo_type="dataset")
        print("uploaded to", a.push_to)


if __name__ == "__main__":
    main()
