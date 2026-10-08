from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

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
            pop.values[slot][:, i] = value


CONTROL = Intervention()


@dataclass
class Inputs:
    signals: np.ndarray  # (n, n_signals) received
    intervention: Intervention = field(default=CONTROL)


@runtime_checkable
class Dynamics(Protocol):
    """Shared rules: a stochastic transition kernel over per-cell state."""

    @property
    def state_schema(self) -> StateSchema: ...

    @property
    def signal_names(self) -> tuple[str, ...]: ...

    def step(
        self, pop: Population, inputs: Inputs, rng: np.random.Generator
    ) -> Population:
        """Must not mutate `pop`."""
        ...

    def emit_signals(self, pop: Population) -> np.ndarray:
        """(n, n_signals) secreted amounts; routing is the simulator's job."""
        ...

    def intervene(self, intervention: Intervention) -> "Dynamics":
        """Effects beyond clamps (which the simulator applies). Usually `self`."""
        ...

    def description_length(self) -> float:
        """Complexity in bits, or a consistent proxy."""
        ...
