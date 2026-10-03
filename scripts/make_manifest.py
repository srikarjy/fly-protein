"""Artifact manifest: assays, model revisions, seeds, budgets, configuration hashes and SHA-256 of every key artifact.

    python scripts/make_manifest.py
Writes results/MANIFEST.json (full) and web/public/data/manifest.json (what the site shows).
"""
import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from embed_assays import ASSAYS  # noqa: E402
from run_benchmark import FLY  # noqa: E402

from flyprotein.bench import adalead, gpbo, methods  # noqa: E402
from flyprotein.bench.data import MODEL  # noqa: E402

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
h = lambda o: hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def hf_rev(repo, kind="model"):
    snaps = Path.home() / ".cache/huggingface/hub" / f"{kind}s--{repo.replace('/', '--')}" / "refs/main"
    return snaps.read_text().strip() if snaps.exists() else None


def pkg(n):
    try:
        return metadata.version(n)
    except metadata.PackageNotFoundError:
        return None


assays = []
for a in ASSAYS:
    v = json.load(open(ROOT / f"data/cache/{a}.variants.json"))
    assays.append(dict(id=a, n_variants=len(v["mutants"]), wt_length=len(v["wt"]), embedding_sha256=sha(ROOT / f"data/cache/{a}.{MODEL}.emb.npy")))

CONFIGS = {
    "random_search": {},
    "random_walk": dict(episode_len=methods.EPISODE_LEN),
    "greedy_local": dict(rule="steepest ascent over the 15 graph neighbours, random restart at a local optimum"),
    "regevo": dict(pop_size=20, sample=5),
    "gpbo_ei": dict(n_init=gpbo.N_INIT, pca_dim=50, kernel="matern52", ell_grid=gpbo.GRID_L, noise_grid=gpbo.GRID_S2, refit_every=gpbo.REFIT_EVERY, acquisition="EI"),
    "adalead_graph": dict(kappa=adalead.KAPPA, batch=adalead.BATCH, climb=adalead.CLIMB, note="graph-adapted, not the published implementation"),
    **{k: {**dict(T=0.1, lr=0.1, scale=0.45, replay=0, memory=False, episode_len=methods.EPISODE_LEN), **v} for k, v in FLY.items()},
    "zero_shot": dict(models=["esm2_t6_8M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D"], modes=["wt-marginal", "masked-marginal (8M, 150M)"], uses_observed_fitness=False),
}
files = ["results/benchmark/queries.npz", "results/benchmark/summary.json", "results/benchmark/tables.md", "results/benchmark/per_run_metrics.csv", "results/ablation/raw.csv", "results/learner/final.csv",
         "results/transfer/raw_scale0.45.csv", "results/hemibrain/raw.csv", "web/public/data/demo_trace.json", "web/public/data/results.json"]
models = {m: dict(repo=f"facebook/{m}", revision=hf_rev(f"facebook/{m}")) for m in ("esm2_t6_8M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D")}
M = dict(
    project="fly-protein", dataset=dict(name="ProteinGym DMS substitutions", repo="OATML-Markslab/ProteinGym_v1", revision=hf_rev("OATML-Markslab/ProteinGym_v1", "dataset")),
    assays=assays, models=models, embedding_used_by_search_methods=MODEL, seeds=list(range(20)), budgets=[50, 100, 200, 500],
    start_rule="start variant = numpy.random.default_rng(1000 + seed).integers(n_variants); method RNG = default_rng((seed, crc32(method_name)))",
    configs={k: dict(config=v, sha256=h(v)) for k, v in CONFIGS.items()},
    artifacts={f: dict(sha256=sha(ROOT / f), bytes=(ROOT / f).stat().st_size) for f in files},
    environment=dict(python=platform.python_version(), platform=platform.platform(), packages={n: pkg(n) for n in ("numpy", "scipy", "pandas", "torch", "transformers", "umap-learn", "neuprint-python", "huggingface_hub")}),
    replay="python scripts/verify_replay.py   # re-runs benchmark jobs and compares with results/benchmark/queries.npz",
)
(ROOT / "results/MANIFEST.json").write_text(json.dumps(M, indent=1))
slim = {k: M[k] for k in ("dataset", "assays", "models", "seeds", "budgets", "start_rule", "configs", "artifacts", "environment", "replay")}
slim["assays"] = [{k: v for k, v in a.items() if k != "embedding_sha256"} | {"embedding_sha256": a["embedding_sha256"][:16]} for a in assays]
(ROOT / "web/public/data/manifest.json").write_text(json.dumps(slim, separators=(",", ":")))
print(json.dumps({k: M[k] for k in ("dataset", "models")}, indent=1))
print({a["id"]: a["n_variants"] for a in assays})
