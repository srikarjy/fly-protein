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
python scripts/demo.py --esm --device cpu           # ESM-2 (8M) embeddings from Hugging Face
python scripts/real_families.py --esm --device cpu  # real proteins (needs internet)
```

Use a fresh venv. In a large Anaconda environment, importing torch and transformers hung on an Abseil mutex message.

## Results so far

Synthetic benchmark: 100 random protein families, 3 members each, queries mutated 30% from the family root, top-1 family retrieval.

| Embedding | Method | Top-1 |
|---|---|---|
| k-mer (420 d) | raw cosine | 95% |
| k-mer (420 d) | FlyHash, default wiring (2000 KCs, 5% active) | 61% |
| k-mer (420 d) | FlyHash, best of a small sweep (8000 KCs, 10% active) | 70% |

Same benchmark with ESM-2 8M (320 d, mean-pooled, CPU):

| Embedding | Method | Top-1 |
|---|---|---|
| ESM-2 8M | raw cosine | 19% |
| ESM-2 8M | FlyHash, default wiring | 18% |

Both ESM-2 numbers are low because this benchmark uses random sequences, which ESM-2 was not trained on, and raw cosine is not mean-centered while FlyHash is. It says little about ESM-2 on natural proteins. `scripts/real_families.py` runs the fair version: real UniProt proteins grouped by Pfam family, with a mean-centered cosine baseline. It has not been run yet.

On k-mer vectors the fly hash is less accurate than exact cosine. Its pitch is a compact binary code that is cheap to store and compare, so the open question is how it compares to cosine and to other hashes on real protein families.

## Status

| Piece | State |
|---|---|
| `FlyHash` with random sparse PN->KC wiring | tested |
| `FlyHash.from_edges` (real synapse counts as weights) | tested on toy edges |
| `KmerEmbedder` | tested |
| `ESM2Embedder` | runs on CPU (MacBook, venv) |
| `scripts/real_families.py` | evaluation logic tested on synthetic families; UniProt download not yet run |
| `connectome.fetch_pn_kc_edges` (hemibrain via neuPrint) | written, not yet run against the server |

## Layout

- `src/flyprotein/flyhash.py` expansion circuit
- `src/flyprotein/embed.py` k-mer and ESM-2 embedders
- `src/flyprotein/search.py` fingerprint index
- `src/flyprotein/connectome.py` hemibrain PN->KC extraction
- `docs/ROADMAP.md` next milestones

## Fly Walker (interactive page)

```bash
pip install -e ".[dev]" pandas pyarrow umap-learn matplotlib
hf download OATML-Markslab/ProteinGym_v1 --repo-type dataset --include "DMS_substitutions/*" --local-dir data/proteingym
python prep/build_data.py --device cpu     # writes web/public/data.json
cd web && npm install && npm run dev
```

See `web/README.md`. The assay is singles-only TEM-1 (no TEM-1 assay in ProteinGym has doubles).
