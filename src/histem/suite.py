import numpy as np
import torch
from pydantic import Field

from histem.data import Dataset
from histem.rng import seeded
from histem.spec import FrozenSpec, Spec, Tagged
from histem.system import CellSystem


class Scores(FrozenSpec):
    fit: dict[tuple[str, str], float]  # (dataset, condition) -> loss
    objective: float

    def by_dataset(self) -> dict[str, float]:
        out: dict[str, list[float]] = {}
        for (ds, _), v in self.fit.items():
            out.setdefault(ds, []).append(v)
        return {k: float(np.mean(v)) for k, v in out.items()}


class Suite(Spec):
    datasets: list[Tagged[Dataset]] = Field(default_factory=list)
    # dataset name -> weight in the objective; missing names weigh 1
    weights: dict[str, float] = Field(default_factory=dict)
    cells_per_condition: int = Field(300, gt=0)
    # `loss` uses this many random conditions per dataset; None means all
    loss_conditions: int | None = Field(None, gt=0)
    # `evaluate` draws with this seed, so score differences aren't sampling noise
    seed: int = 0

    def loss(self, system: CellSystem) -> torch.Tensor:
        """Differentiable objective, on fresh samples."""
        if not self.datasets:
            raise ValueError("suite has no datasets")
        per_ds = {
            ds.name: torch.stack(
                [
                    ds.loss(system, c, self.cells_per_condition)
                    for c in self._pick_conditions(ds.conditions)
                ]
            ).mean()
            for ds in self.datasets
        }
        w = {k: self.weights.get(k, 1.0) for k in per_ds}
        total = torch.stack([w[k] * v for k, v in per_ds.items()]).sum()
        return total / sum(w.values())

    def _pick_conditions(self, conditions: list[str]) -> list[str]:
        if self.loss_conditions is None or self.loss_conditions >= len(conditions):
            return conditions
        idx = torch.randperm(len(conditions))[: self.loss_conditions]
        return [conditions[i] for i in idx.tolist()]

    def evaluate(self, system: CellSystem) -> Scores:
        fit = {}
        with torch.no_grad(), seeded(self.seed):
            for ds in self.datasets:
                for cond in ds.conditions:
                    loss = ds.loss(system, cond, self.cells_per_condition)
                    fit[(ds.name, cond)] = float(loss)
        if not fit:
            raise ValueError("suite has no (dataset, condition) pairs to score")
        per_ds = Scores(fit=fit, objective=0.0).by_dataset()
        w = {k: self.weights.get(k, 1.0) for k in per_ds}
        objective = sum(w[k] * v for k, v in per_ds.items()) / sum(w.values())
        return Scores(fit=fit, objective=objective)
