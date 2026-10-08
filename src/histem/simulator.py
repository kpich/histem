"""Run a population of cells forward under shared Dynamics.

Signaling is routed here, not inside Dynamics, so the same rules can run in a dish
(well-mixed, no neighbor graph) or in a tissue patch (spatial neighbor graph).
Received signal = autocrine * own + paracrine * neighbor mean
                + endocrine * population mean.
"""

from typing import Self

import numpy as np
import scipy.sparse as sp
from pydantic import Field, model_validator
from scipy.spatial import cKDTree

from histem.dynamics import CONTROL, Dynamics, Inputs, Intervention
from histem.spec import FrozenSpec
from histem.state import Population


class Signaling(FrozenSpec):
    autocrine: float = Field(1.0, ge=0)
    paracrine: float = Field(1.0, ge=0)
    endocrine: float = Field(0.0, ge=0)
    # (n, n) adjacency; None = no paracrine routing
    neighbors: sp.csr_matrix | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (
            self.neighbors is not None
            and self.neighbors.shape[0] != (self.neighbors.shape[1])
        ):
            raise ValueError(f"neighbors must be square, got {self.neighbors.shape}")
        return self

    def route(self, emitted: np.ndarray) -> np.ndarray:
        received = self.autocrine * emitted
        if self.neighbors is not None and self.paracrine:
            if self.neighbors.shape[0] != emitted.shape[0]:
                raise ValueError("neighbor graph size does not match population")
            deg = np.asarray(self.neighbors.sum(axis=1)).clip(min=1)
            received = received + self.paracrine * (self.neighbors @ emitted) / deg
        if self.endocrine:
            received = received + self.endocrine * emitted.mean(axis=0, keepdims=True)
        return received


def knn_graph(positions: np.ndarray, k: int = 6) -> sp.csr_matrix:
    idx = np.asarray(cKDTree(positions).query(positions, k=k + 1)[1])
    rows = np.repeat(np.arange(len(positions)), k)
    return sp.csr_matrix(
        (np.ones(rows.size), (rows, idx[:, 1:].ravel())), shape=(len(positions),) * 2
    )


def simulate(
    dynamics: Dynamics,
    pop: Population,
    steps: int,
    rng: np.random.Generator,
    *,
    intervention: Intervention = CONTROL,
    signaling: Signaling | None = None,
    record_every: int | None = None,
) -> tuple[Population, list[Population]]:
    """Returns (final population, snapshots taken every `record_every` steps)."""
    signaling = signaling or Signaling()
    dyn = dynamics.intervene(intervention)
    pop = pop.copy()
    intervention.apply(pop)
    trajectory = []
    for t in range(steps):
        received = signaling.route(dyn.emit_signals(pop))
        pop = dyn.step(pop, Inputs(received, intervention), rng)
        intervention.apply(pop)
        if record_every and (t + 1) % record_every == 0:
            trajectory.append(pop.copy())
    return pop, trajectory
