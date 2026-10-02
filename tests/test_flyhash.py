import numpy as np

from flyprotein import FlyHash, FingerprintIndex, KmerEmbedder

AA = "ACDEFGHIKLMNPQRSTVWY"


def mutate(seq, frac, rng):
    s = list(seq)
    for i in rng.choice(len(s), size=int(frac * len(s)), replace=False):
        s[i] = AA[rng.integers(20)]
    return "".join(s)


def test_sparsity_is_exact():
    X = np.random.default_rng(0).normal(size=(10, 50))
    fh = FlyHash(d_in=50, n_kc=2000, active_frac=0.05).fit(X)
    fp = fh.encode(X)
    assert fp.shape == (10, 2000)
    assert (fp.sum(axis=1) == 100).all()


def test_similar_inputs_share_more_kcs_than_random():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 50))
    fh = FlyHash(d_in=50).fit(X)
    a = X[0]
    near = a + 0.1 * rng.normal(size=50)
    far = X[1]
    fa, fn, ff = fh.encode(np.stack([a, near, far]))
    assert (fa & fn).sum() > 3 * (fa & ff).sum()


def test_from_edges_matches_weights():
    fh = FlyHash.from_edges([(0, 0, 5), (1, 0, 2), (1, 1, 7)], n_pn=2, n_kc=2, active_frac=0.5)
    assert fh.W[0, 0] == 5 and fh.W[0, 1] == 2 and fh.W[1, 1] == 7


def test_protein_family_retrieval():
    rng = np.random.default_rng(2)
    bases = ["".join(rng.choice(list(AA), size=200)) for _ in range(30)]
    emb = KmerEmbedder()
    members = [mutate(b, 0.1, rng) for b in bases for _ in range(3)]
    labels = [i for i in range(30) for _ in range(3)]
    queries = [mutate(b, 0.1, rng) for b in bases]
    fh = FlyHash(d_in=emb.dim).fit(emb(members))
    index = FingerprintIndex(fh.encode(emb(members)), labels)
    hits = index.query(fh.encode(emb(queries)), k=1)
    acc = np.mean([h[0][0] == i for i, h in enumerate(hits)])
    assert acc >= 0.9
