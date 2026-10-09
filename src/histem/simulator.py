from typing import Self

import torch
from pydantic import Field, model_validator

from histem.dynamics import CONTROL, Dynamics, Inputs, Intervention
from histem.spec import FrozenSpec, Tensor
from histem.state import Population


class Signaling(FrozenSpec):
    autocrine: float = Field(1.0, ge=0)
    paracrine: float = Field(1.0, ge=0)
    endocrine: float = Field(0.0, ge=0)
    neighbors: Tensor | None = None  # (2, n_edges) of (receiver, sender)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.neighbors is not None and (
            self.neighbors.ndim != 2 or self.neighbors.shape[0] != 2
        ):
            raise ValueError(
                f"neighbors must have shape (2, n_edges), got {self.neighbors.shape}"
            )
        return self

    def route(self, emitted: torch.Tensor) -> torch.Tensor:
        """autocrine * own + paracrine * neighbor mean + endocrine * global mean."""
        received = self.autocrine * emitted
        if self.neighbors is not None and self.paracrine:
            n = emitted.shape[0]
            if int(self.neighbors.max()) >= n:
                raise ValueError("neighbor graph size does not match population")
            rcv, snd = self.neighbors.to(emitted.device)
            total = torch.zeros_like(emitted).index_add_(0, rcv, emitted[snd])
            deg = torch.bincount(rcv, minlength=n).clamp(min=1).unsqueeze(1)
            received = received + self.paracrine * total / deg
        if self.endocrine:
            received = received + self.endocrine * emitted.mean(dim=0, keepdim=True)
        return received


def knn_graph(positions: torch.Tensor, k: int = 6) -> torch.Tensor:
    """(2, n * k) edges from each cell to its k nearest neighbors."""
    d = torch.cdist(positions, positions)
    d.fill_diagonal_(float("inf"))
    idx = d.topk(k, largest=False).indices
    rcv = torch.arange(len(positions), device=positions.device).repeat_interleave(k)
    return torch.stack([rcv, idx.reshape(-1)])


def simulate(
    dynamics: Dynamics,
    pop: Population,
    steps: int,
    *,
    intervention: Intervention = CONTROL,
    signaling: Signaling | None = None,
    record_every: int | None = None,
) -> tuple[Population, list[Population]]:
    """Returns (final population, snapshots every `record_every` steps)."""
    signaling = signaling or Signaling()
    dyn = dynamics.intervene(intervention)
    pop = pop.copy()
    intervention.apply(pop)
    trajectory = []
    for t in range(steps):
        received = signaling.route(dyn.emit_signals(pop))
        pop = dyn.step(pop, Inputs(received, intervention))
        intervention.apply(pop)
        if record_every and (t + 1) % record_every == 0:
            trajectory.append(pop.copy())
    return pop, trajectory
