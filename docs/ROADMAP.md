# Roadmap

## M1: Smell a protein (in progress)
- [x] FlyHash fingerprinting + retrieval harness
- [ ] Run with ESM-2 embeddings on a real family set (Pfam/UniProt subset)
- [ ] Swap random wiring for real hemibrain PN->KC counts and compare
- [ ] Baselines: SimHash, random-projection LSH at equal code length

## M2: Learn good vs bad proteins
Mushroom body learner: Kenyon cells -> output neurons, with dopamine-gated plasticity as the reward signal. Train on a binary protein label (stable/unstable, binder/non-binder) and compare against logistic regression on the same fingerprints.

## M3: Chase the best protein
Treat a fitness score (ESM pseudo-likelihood, then Boltz binding) as an odor concentration over a 2D map of embedding space. Drive a central-complex steering model uphill and read out the path as a proposed variant trajectory.

## Later
- NVIDIA side: cuGraph for connectome graph analytics, BioNeMo NIMs (ProteinMPNN, OpenFold, DiffDock) as generators and scorers
- Interactive viewer for the fly walking the landscape
