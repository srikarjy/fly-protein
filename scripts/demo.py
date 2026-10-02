"""Protein family retrieval: FlyHash fingerprints vs raw cosine similarity.

python scripts/demo.py                  # k-mer embeddings, numpy only
python scripts/demo.py --esm            # ESM-2 embeddings (torch + transformers)
"""
import argparse

import numpy as np

from flyprotein import ESM2Embedder, FingerprintIndex, FlyHash, KmerEmbedder

AA = "ACDEFGHIKLMNPQRSTVWY"


def mutate(seq, frac, rng):
    s = list(seq)
    for i in rng.choice(len(s), size=int(frac * len(s)), replace=False):
        s[i] = AA[rng.integers(20)]
    return "".join(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--esm", action="store_true")
    p.add_argument("--families", type=int, default=100)
    p.add_argument("--mut", type=float, default=0.3)
    a = p.parse_args()

    rng = np.random.default_rng(0)
    bases = ["".join(rng.choice(list(AA), size=250)) for _ in range(a.families)]
    members = [mutate(b, a.mut, rng) for b in bases for _ in range(3)]
    labels = [i for i in range(a.families) for _ in range(3)]
    queries = [mutate(b, a.mut, rng) for b in bases]

    emb = ESM2Embedder() if a.esm else KmerEmbedder()
    M, Q = emb(members), emb(queries)

    cos = (Q / np.linalg.norm(Q, axis=1, keepdims=True)) @ (M / np.linalg.norm(M, axis=1, keepdims=True)).T
    cos_acc = np.mean([labels[j] == i for i, j in enumerate(cos.argmax(1))])

    fh = FlyHash(d_in=emb.dim).fit(M)
    idx = FingerprintIndex(fh.encode(M), labels)
    fly_acc = np.mean([h[0][0] == i for i, h in enumerate(idx.query(fh.encode(Q), k=1))])

    print(f"embedder: {'ESM-2' if a.esm else 'k-mer'} ({emb.dim} dims), {a.families} families, {a.mut:.0%} mutated")
    print(f"raw cosine top-1:   {cos_acc:.2%}")
    print(f"FlyHash top-1:      {fly_acc:.2%}  (2000 KCs, 5% active)")


if __name__ == "__main__":
    main()
