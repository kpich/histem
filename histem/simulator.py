"""Run a population of cells forward under shared Dynamics.

Signaling is routed here, not inside Dynamics, so the same rules can run in a dish
(well-mixed, no neighbor graph) or in a tissue patch (spatial neighbor graph).
Received signal = autocrine * own + paracrine * mean over neighbors + endocrine * population mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from histem.dynamics import CONTROL, Dynamics, Inputs, Intervention
from histem.state import Population


@dataclass
class Signaling:
    autocrine: float = 1.0
    paracrine: float = 1.0
    endocrine: float = 0.0
    neighbors: sp.csr_matrix | None = None  # (n, n) adjacency; None = no paracrine routing

    def route(self, emitted: np.ndarray) -> np.ndarray:
        received = self.autocrine * emitted
        if self.neighbors is not None and self.paracrine:
            deg = np.asarray(self.neighbors.sum(axis=1)).clip(min=1)
            received = received + self.paracrine * (self.neighbors @ emitted) / deg
        if self.endocrine:
            received = received + self.endocrine * emitted.mean(axis=0, keepdims=True)
        return received


def knn_graph(positions: np.ndarray, k: int = 6) -> sp.csr_matrix:
    from scipy.spatial import cKDTree

    _, idx = cKDTree(positions).query(positions, k=k + 1)
    rows = np.repeat(np.arange(len(positions)), k)
    return sp.csr_matrix((np.ones(rows.size), (rows, idx[:, 1:].ravel())), shape=(len(positions),) * 2)


def simulate(
    dynamics: Dynamics,
    pop: Population,
    steps: int,
    rng: np.random.Generator,
    intervention: Intervention = CONTROL,
    signaling: Signaling | None = None,
    record_every: int | None = None,
) -> tuple[Population, list[Population]]:
    """Returns (final population, trajectory snapshots taken every `record_every` steps)."""
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
