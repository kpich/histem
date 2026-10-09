"""Continuous dynamics as an Euler-Maruyama SDE with an MLP drift:

    x <- x + dt * drift(x, in) + exp(log_noise) * sqrt(dt) * eps

Every slot must be continuous. Emitted signals are softplus(linear(x)).
"""

import math
from collections.abc import Iterator
from typing import Self

import torch
from pydantic import Field, PrivateAttr, model_validator
from torch import nn

from histem.dynamics import Inputs, Intervention
from histem.spec import FrozenSpec, Tensor
from histem.state import Population, StateSchema


class _Net(nn.Module):
    def __init__(self, dim: int, n_signals: int, hidden: int):
        super().__init__()
        self.drift = nn.Sequential(
            nn.Linear(dim + n_signals, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        self.emit = nn.Linear(dim, n_signals) if n_signals else None
        self.log_noise = nn.Parameter(torch.full((dim,), -1.0))


class NeuralDynamics(FrozenSpec):
    state_schema: StateSchema
    signals: tuple[str, ...] = ()
    hidden: int = Field(64, gt=0)
    dt: float = Field(0.1, gt=0)
    # empty means freshly initialized; afterwards it holds the live parameters
    weights: dict[str, Tensor] = Field(default_factory=dict)

    _net: _Net = PrivateAttr()

    @model_validator(mode="after")
    def _build(self) -> Self:
        if bad := [s.name for s in self.state_schema.slots if s.kind != "continuous"]:
            raise ValueError(f"NeuralDynamics needs continuous slots, got {bad}")
        dim = len(self.state_schema.variables)
        self._net = _Net(dim, len(self.signals), self.hidden)
        if self.weights:
            self._net.load_state_dict(self.weights, assign=True)
        self.weights.clear()
        self.weights.update(self._net.named_parameters())
        return self

    @property
    def signal_names(self) -> tuple[str, ...]:
        return self.signals

    def parameters(self) -> Iterator[torch.Tensor]:
        return iter(self.weights.values())

    def _state(self, pop: Population) -> torch.Tensor:
        return torch.cat([pop.values[s.name] for s in self.state_schema.slots], dim=1)

    def step(self, pop: Population, inputs: Inputs) -> Population:
        x = self._state(pop)
        drift = self._net.drift(torch.cat([x, inputs.signals], dim=1))
        noise = self._net.log_noise.exp() * math.sqrt(self.dt) * torch.randn_like(x)
        x = x + self.dt * drift + noise
        dims = [s.dim for s in self.state_schema.slots]
        values = dict(
            zip(
                [s.name for s in self.state_schema.slots],
                x.split(dims, dim=1),
                strict=True,
            )
        )
        return Population(self.state_schema, values, pop.positions, dict(pop.meta))

    def emit_signals(self, pop: Population) -> torch.Tensor:
        if self._net.emit is None:
            return torch.zeros((pop.n, 0), device=pop.device)
        return nn.functional.softplus(self._net.emit(self._state(pop)))

    def intervene(self, intervention: Intervention) -> Self:
        return self

    def description_length(self) -> float:
        """32 bits per parameter."""
        return float(sum(p.numel() for p in self.weights.values())) * 32.0
