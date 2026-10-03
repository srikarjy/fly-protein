"""The only door to experimental fitness.

Methods receive an Oracle, never the fitness array. A query reveals one variant's measured fitness and
costs one unit the first time that variant is queried; re-reading an already-queried variant is free
(it is a lookup of a measurement we already own). When the budget is spent the next *new* query raises
BudgetExhausted. The ordered query log is the deterministic record of a run; all metrics are computed
from it afterwards by the evaluator, which is the only code that sees the full landscape.
"""
from __future__ import annotations

import numpy as np


class BudgetExhausted(Exception):
    pass


class Oracle:
    def __init__(self, fitness: np.ndarray, budget: int):
        self._f = np.asarray(fitness, dtype=np.float64)
        self.budget = int(budget)
        self.log: list[int] = []
        self._seen: dict[int, float] = {}

    def query(self, i) -> float:
        i = int(i)
        if i in self._seen:
            return self._seen[i]
        if len(self.log) >= self.budget:
            raise BudgetExhausted
        self.log.append(i)
        self._seen[i] = float(self._f[i])
        return self._seen[i]

    def queried(self, i) -> bool:
        return int(i) in self._seen

    @property
    def n_queries(self) -> int:
        return len(self.log)

    def observed(self) -> tuple[np.ndarray, np.ndarray]:
        """(indices, fitness) of everything queried so far, in query order."""
        idx = np.array(self.log, dtype=np.int64)
        return idx, np.array([self._seen[i] for i in self.log], dtype=np.float64)
