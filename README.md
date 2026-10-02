# fly-protein

Use circuits from the Drosophila connectome to do protein tasks.

Milestone 1 (this commit): the fly olfactory circuit as a protein fingerprinting and similarity-search layer.

```
sequence -> embedding (k-mer or ESM-2) -> PN->KC expansion -> top-5% winner-take-all -> sparse binary fingerprint
```

## Quick start

```bash
pip install -e ".[dev]"
pytest
python scripts/demo.py                 # numpy-only k-mer embeddings
pip install -e ".[esm]"
python scripts/demo.py --esm           # ESM-2 (8M) embeddings from Hugging Face
```

## Results so far

Synthetic benchmark: 100 random protein families, 3 members each, queries mutated 30% from the family root, top-1 family retrieval.

| Embedding | Method | Top-1 |
|---|---|---|
| k-mer (420 d) | raw cosine | 95% |
| k-mer (420 d) | FlyHash, default wiring (2000 KCs, 5% active) | 61% |
| k-mer (420 d) | FlyHash, best of a small sweep (8000 KCs, 10% active) | 70% |

On k-mer vectors the fly hash is less accurate than exact cosine. Its pitch is a compact binary code that is cheap to store and compare, so the open question is how it compares to cosine and to other hashes on ESM-2 embeddings and real protein families. That run has not happened yet (the cloud workspace cannot reach the Hugging Face Hub).

## Status

| Piece | State |
|---|---|
| `FlyHash` with random sparse PN->KC wiring | tested |
| `FlyHash.from_edges` (real synapse counts as weights) | tested on toy edges |
| `KmerEmbedder` | tested |
| `ESM2Embedder` | written, not yet run |
| `connectome.fetch_pn_kc_edges` (hemibrain via neuPrint) | written, not yet run against the server |

## Layout

- `src/flyprotein/flyhash.py` expansion circuit
- `src/flyprotein/embed.py` k-mer and ESM-2 embedders
- `src/flyprotein/search.py` fingerprint index
- `src/flyprotein/connectome.py` hemibrain PN->KC extraction
- `docs/ROADMAP.md` next milestones
