import numpy as np
import torch
from pydantic import Field

from histem.data import Dataset
from histem.rng import seeded
from histem.spec import FrozenSpec, Spec
from histem.system import CellSystem


class Scores(FrozenSpec):
    fit: dict[tuple[str, str], float]  # (dataset, condition) -> loss
    description_length: float
    objective: float

    def by_dataset(self) -> dict[str, float]:
        out: dict[str, list[float]] = {}
        for (ds, _), v in self.fit.items():
            out.setdefault(ds, []).append(v)
        return {k: float(np.mean(v)) for k, v in out.items()}


class Suite(Spec):
    datasets: list[Dataset] = Field(default_factory=list)
    # dataset name -> weight in the objective; missing names weigh 1
    weights: dict[str, float] = Field(default_factory=dict)
    complexity_weight: float = Field(1e-3, ge=0)
    cells_per_condition: int = Field(300, gt=0)
    # same seed for every candidate, so score differences aren't sampling noise
    seed: int = 0

    def add(self, dataset: Dataset) -> None:
        if any(d.name == dataset.name for d in self.datasets):
            raise ValueError(f"dataset {dataset.name!r} already in suite")
        self.datasets.append(dataset)

    def evaluate(self, system: CellSystem) -> Scores:
        fit = {}
        with torch.no_grad(), seeded(self.seed):
            for ds in self.datasets:
                for cond in ds.conditions:
                    loss = ds.loss(system, cond, self.cells_per_condition)
                    fit[(ds.name, cond)] = float(loss)
        if not fit:
            raise ValueError("suite has no (dataset, condition) pairs to score")
        scores = Scores(fit=fit, description_length=0.0, objective=0.0)
        per_ds = scores.by_dataset()
        w = {k: self.weights.get(k, 1.0) for k in per_ds}
        fit_term = sum(w[k] * v for k, v in per_ds.items()) / sum(w.values())
        dl = system.description_length()
        return Scores(
            fit=fit,
            description_length=dl,
            objective=fit_term + self.complexity_weight * dl,
        )

    @staticmethod
    def regressions(
        before: Scores, after: Scores, tolerance: float = 0.05
    ) -> list[str]:
        """Datasets whose mean fit got worse by more than `tolerance` (relative)."""
        b, a = before.by_dataset(), after.by_dataset()
        return [k for k in b if k in a and a[k] > b[k] * (1 + tolerance)]
