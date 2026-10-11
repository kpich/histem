from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch

from histem.spec import FrozenSpec
from histem.state import Population, StateSchema


class Intervention(FrozenSpec):
    """`clamps` are re-applied after every step; `tags` are for `Dynamics.intervene`."""

    name: str = "control"
    clamps: tuple[tuple[str, float], ...] = ()
    tags: tuple[str, ...] = ()

    def apply(self, pop: Population) -> None:
        for variable, value in self.clamps:
            slot, i = pop.schema.locate(variable)
            x = pop.values[slot]
            # out-of-place, so it is safe inside an autograd graph
            pop.values[slot] = x.index_fill(
                1, torch.tensor([i], device=x.device), value
            )


CONTROL = Intervention()


@dataclass
class Inputs:
    signals: torch.Tensor  # (n, n_signals) received
    intervention: Intervention = field(default=CONTROL)


@dataclass
class Transitions:
    """Recorded steps of some dynamics, flattened in schema order."""

    x: torch.Tensor  # (n, n_variables) before the step
    received: torch.Tensor  # (n, n_signals)
    emitted: torch.Tensor  # (n, n_signals) by x
    next: torch.Tensor  # (n, n_variables) after the step, before clamps

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def batch(self, idx: torch.Tensor) -> "Transitions":
        return Transitions(
            self.x[idx], self.received[idx], self.emitted[idx], self.next[idx]
        )


@runtime_checkable
class Dynamics(Protocol):
    """Shared rules: a stochastic transition kernel over per-cell state."""

    @property
    def state_schema(self) -> StateSchema: ...

    @property
    def signal_names(self) -> tuple[str, ...]: ...

    def step(self, pop: Population, inputs: Inputs) -> Population:
        """Must not mutate `pop`."""
        ...

    def emit_signals(self, pop: Population) -> torch.Tensor:
        """(n, n_signals) secreted amounts; routing is the simulator's job."""
        ...

    def intervene(self, intervention: Intervention) -> "Dynamics":
        """Effects beyond clamps (which the simulator applies). Usually `self`."""
        ...

    def description_length(self) -> float:
        """Complexity in bits, or a consistent proxy."""
        ...
