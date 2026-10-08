"""The pluggable part: shared rules that move one cell's state forward.

Any representation (logic network, LLM-written code, symbolic ODE, a conditional
discrete-diffusion kernel, a distilled neural emulator, ...) qualifies if it implements
`Dynamics`. The contract is a *stochastic transition kernel*: given current state and
inputs, sample the next state. Unconditional generators do not satisfy this contract.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from histem.spec import FrozenSpec
from histem.state import Population, StateSchema


class Intervention(FrozenSpec):
    """An experimental condition, independent of any particular Dynamics.

    `clamps` pins variables to fixed values after every step (knockout = clamp to 0,
    overexpression = clamp high). `tags` carries anything representation-specific, such
    as a drug name, which a Dynamics may interpret through `Dynamics.intervene`.
    """

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
    """What a cell receives from outside itself during one step."""

    signals: np.ndarray  # (n, n_signals) received ligand levels
    intervention: Intervention = field(default=CONTROL)


@runtime_checkable
class Dynamics(Protocol):
    @property
    def state_schema(self) -> StateSchema: ...

    @property
    def signal_names(self) -> tuple[str, ...]: ...

    def step(
        self, pop: Population, inputs: Inputs, rng: np.random.Generator
    ) -> Population:
        """Sample the next state for every cell. Must not mutate `pop`."""
        ...

    def emit_signals(self, pop: Population) -> np.ndarray:
        """(n, n_signals) secreted ligand amounts.

        Routing between cells is the Simulator's job.
        """
        ...

    def intervene(self, intervention: Intervention) -> "Dynamics":
        """A Dynamics modified by the intervention's representation-specific effects.
        Clamps are applied by the Simulator, so most implementations return self."""
        ...

    def description_length(self) -> float:
        """Complexity in bits (or a consistent proxy); the regulariser in Objective."""
        ...
