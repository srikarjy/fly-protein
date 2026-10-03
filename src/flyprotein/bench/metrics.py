"""Evaluator-side metrics. Percentile = share of the assay's variants with fitness <= the value (0-100]."""
from __future__ import annotations

import numpy as np


def percentile_table(fitness: np.ndarray):
    s = np.sort(np.asarray(fitness))
    return lambda f: 100.0 * np.searchsorted(s, f, side="right") / len(s)


def curve_percentiles(log, fitness: np.ndarray, max_budget: int = 500) -> np.ndarray:
    """best-so-far percentile after b unique queries, b = 1..max_budget (NaN where the run made fewer queries)."""
    pct = percentile_table(fitness)
    out = np.full(max_budget, np.nan)
    if len(log):
        best = np.maximum.accumulate(np.asarray(fitness)[np.asarray(log)[:max_budget]])
        out[: len(best)] = pct(best)
    return out
