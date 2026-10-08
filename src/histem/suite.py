"""Datasets as a regression test suite, and the objective used to score world models.

Lifelong learning: every dataset ever added stays in the suite. A candidate update
to the world model is accepted only if it improves the objective without regressing any
dataset beyond tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from histem.data import Dataset
from histem.metrics import population_distance
from histem.world import WorldModel


@dataclass
class Scores:
    fit: dict[tuple[str, str], float]  # (dataset, condition) -> distance
    description_length: float
    objective: float

    def by_dataset(self) -> dict[str, float]:
        out: dict[str, list[float]] = {}
        for (ds, _), v in self.fit.items():
            out.setdefault(ds, []).append(v)
        return {k: float(np.mean(v)) for k, v in out.items()}


@dataclass
class Suite:
    datasets: list[Dataset] = field(default_factory=list)
    complexity_weight: float = 1e-3  # lambda: the interpretability/fit tradeoff knob
    cells_per_condition: int = 300
    # common random numbers: one seed for every candidate reduces scoring noise
    seed: int = 0

    def add(self, dataset: Dataset) -> None:
        self.datasets.append(dataset)

    def evaluate(self, world: WorldModel) -> Scores:
        rng = np.random.default_rng(self.seed)
        fit = {}
        for ds in self.datasets:
            for cond in ds.conditions:
                observed = ds.cells(cond)
                sim = world.sample(
                    self.cells_per_condition, rng, ds.interventions[cond], ds.modality
                )
                fit[(ds.name, cond)] = population_distance(observed, sim, rng=rng)
        dl = world.description_length()
        objective = float(np.mean(list(fit.values()))) + self.complexity_weight * dl
        return Scores(fit, dl, objective)

    def regressions(
        self, before: Scores, after: Scores, tolerance: float = 0.05
    ) -> list[str]:
        """Datasets whose mean fit got worse by more than `tolerance` (relative)."""
        b, a = before.by_dataset(), after.by_dataset()
        return [k for k in b if k in a and a[k] > b[k] * (1 + tolerance)]
