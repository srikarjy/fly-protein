"""Family retrieval on real proteins (UniProt reviewed entries, grouped by Pfam).

python scripts/real_families.py                          # k-mer embeddings only
python scripts/real_families.py --esm --device cpu       # also ESM-2

Needs network access to rest.uniprot.org on the first run. Sequences are
cached in data/ (gitignored). Compares raw cosine, mean-centered cosine (the
fair baseline for FlyHash, which also subtracts the mean) and FlyHash.
"""
import argparse
import os
import urllib.parse
import urllib.request

import numpy as np

from flyprotein import ESM2Embedder, FingerprintIndex, FlyHash, KmerEmbedder

FAMILIES = {
    "PF00069": "protein kinase",
    "PF00067": "cytochrome P450",
    "PF00076": "RNA recognition motif",
    "PF00096": "zinc finger C2H2",
    "PF00071": "Ras GTPase",
    "PF00018": "SH3 domain",
    "PF00072": "response regulator",
    "PF00005": "ABC transporter",
    "PF00001": "7TM GPCR",
    "PF00106": "short-chain dehydrogenase",
}


def parse_fasta(text):
    seqs = []
    for rec in text.split(">")[1:]:
        lines = rec.strip().split("\n")
        seq = "".join(lines[1:]).strip()
        if seq:
            seqs.append(seq)
    return seqs


def fetch_family(pfam_id, n, cache_dir="data"):
    path = os.path.join(cache_dir, f"{pfam_id}.fasta")
    if not os.path.exists(path):
        q = urllib.parse.quote(f"(xref:pfam-{pfam_id}) AND (reviewed:true)")
        url = f"https://rest.uniprot.org/uniprotkb/search?format=fasta&size={n}&query={q}"
        with urllib.request.urlopen(url, timeout=60) as r:
            text = r.read().decode()
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
    with open(path) as f:
        return parse_fasta(f.read())


def _cos(A, B):
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    return A @ B.T


def evaluate(families, emb, seed=0, max_len=400, query_frac=1 / 3):
    """families: {name: [sequences]}. Returns top-1 family accuracy per method."""
    rng = np.random.default_rng(seed)
    idx_seqs, idx_lab, q_seqs, q_lab = [], [], [], []
    for fam, seqs in families.items():
        seqs = [s[:max_len] for s in seqs]
        order = rng.permutation(len(seqs))
        n_q = max(1, int(len(seqs) * query_frac))
        for k, i in enumerate(order):
            if k < n_q:
                q_seqs.append(seqs[i])
                q_lab.append(fam)
            else:
                idx_seqs.append(seqs[i])
                idx_lab.append(fam)
    M, Q = emb(idx_seqs), emb(q_seqs)
    mu = M.mean(axis=0)

    def acc(top):
        return float(np.mean([idx_lab[j] == q_lab[i] for i, j in enumerate(top)]))

    fh = FlyHash(d_in=emb.dim).fit(M)
    hits = FingerprintIndex(fh.encode(M), idx_lab).query(fh.encode(Q), k=1)
    return {
        "cosine": acc(_cos(Q, M).argmax(1)),
        "cosine_centered": acc(_cos(Q - mu, M - mu).argmax(1)),
        "flyhash": float(np.mean([h[0][0] == q_lab[i] for i, h in enumerate(hits)])),
        "n_index": len(idx_seqs),
        "n_query": len(q_seqs),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--esm", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--per-family", type=int, default=40)
    a = p.parse_args()

    fams = {}
    for pf, name in FAMILIES.items():
        fams[pf] = fetch_family(pf, a.per_family)
        print(f"{pf} {name}: {len(fams[pf])} sequences")

    embedders = {"k-mer": KmerEmbedder()}
    if a.esm:
        embedders["ESM-2 8M"] = ESM2Embedder(device=a.device)
    print(f"\n{'embedding':10s} {'cosine':>8s} {'centered':>9s} {'FlyHash':>8s}")
    for name, emb in embedders.items():
        r = evaluate(fams, emb)
        print(f"{name:10s} {r['cosine']:8.1%} {r['cosine_centered']:9.1%} {r['flyhash']:8.1%}   (index {r['n_index']}, queries {r['n_query']})")


if __name__ == "__main__":
    main()
