"""Protein sequence -> vector. Two backends with the same call signature."""
from __future__ import annotations

from itertools import product

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"


class KmerEmbedder:
    """Numpy-only baseline: 1-mer + 2-mer composition (20 + 400 = 420 dims)."""

    def __init__(self):
        self.index = {a: i for i, a in enumerate(AA)}
        pairs = ["".join(p) for p in product(AA, repeat=2)]
        self.pair_index = {p: 20 + i for i, p in enumerate(pairs)}
        self.dim = 20 + len(pairs)

    def __call__(self, seqs: list[str]) -> np.ndarray:
        out = np.zeros((len(seqs), self.dim), dtype=np.float32)
        for r, s in enumerate(seqs):
            s = "".join(c for c in s.upper() if c in self.index)
            for c in s:
                out[r, self.index[c]] += 1
            for i in range(len(s) - 1):
                out[r, self.pair_index[s[i : i + 2]]] += 1
            n = np.linalg.norm(out[r])
            if n > 0:
                out[r] /= n
        return out


class ESM2Embedder:
    """Mean-pooled ESM-2 embeddings via Hugging Face transformers.

    Needs `pip install torch transformers`. Weights download from the Hub on
    first use. Default is the 8M model (320 dims), small enough for a laptop.
    """

    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D", device: str | None = None, batch_size: int = 8):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.batch_size = batch_size
        self.dim = self.model.config.hidden_size

    def __call__(self, seqs: list[str]) -> np.ndarray:
        torch = self.torch
        chunks = []
        for i in range(0, len(seqs), self.batch_size):
            enc = self.tok(seqs[i : i + self.batch_size], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                h = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].clone()
            lengths = mask.sum(1)
            mask[:, 0] = 0  # CLS
            mask[torch.arange(mask.shape[0]), lengths - 1] = 0  # EOS
            m = mask.unsqueeze(-1).to(h.dtype)
            chunks.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy())
        return np.concatenate(chunks, axis=0)
