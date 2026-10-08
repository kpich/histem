"""Datasets as a regression test suite, and the objective used to score world models.

Lifelong learning: every dataset ever added stays in the suite. A candidate update
to the world model is accepted only if it improves the objective without regressing any
dataset beyond tolerance.
"""

import numpy as np
from pydantic import Field

from histem.data import Dataset
from histem.metrics import population_distance
from histem.spec import FrozenSpec, Spec
from histem.world import WorldModel


class Scores(FrozenSpec):
    fit: dict[tuple[str, str], float]  # (dataset, condition) -> distance
    description_length: float
    objective: float

    def by_dataset(self) -> dict[str, float]:
        out: dict[str, list[float]] = {}
        for (ds, _), v in self.fit.items():
            out.setdefault(ds, []).append(v)
        return {k: float(np.mean(v)) for k, v in out.items()}


class Suite(Spec):
    datasets: list[Dataset] = Field(default_factory=list)
    # lambda: the interpretability/fit tradeoff knob
    complexity_weight: float = Field(1e-3, ge=0)
    cells_per_condition: int = Field(300, gt=0)
    # common random numbers: one seed for every candidate reduces scoring noise
    seed: int = 0

    def add(self, dataset: Dataset) -> None:
        if any(d.name == dataset.name for d in self.datasets):
            raise ValueError(f"dataset {dataset.name!r} already in suite")
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
        if not fit:
            raise ValueError("suite has no (dataset, condition) pairs to score")
        dl = world.description_length()
        objective = float(np.mean(list(fit.values()))) + self.complexity_weight * dl
        return Scores(fit=fit, description_length=dl, objective=objective)

    @staticmethod
    def regressions(
        before: Scores, after: Scores, tolerance: float = 0.05
    ) -> list[str]:
        """Datasets whose mean fit got worse by more than `tolerance` (relative)."""
        b, a = before.by_dataset(), after.by_dataset()
        return [k for k in b if k in a and a[k] > b[k] * (1 + tolerance)]
