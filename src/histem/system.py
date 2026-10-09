from typing import Protocol, runtime_checkable

import torch
from pydantic import Field

from histem.dynamics import CONTROL, Dynamics, Intervention
from histem.observers import Observer
from histem.simulator import Signaling, simulate
from histem.spec import FrozenSpec, Tagged
from histem.state import Population, StateSchema


@runtime_checkable
class Init(Protocol):
    """Prior over initial states; it decides the device the system runs on."""

    def sample(self, schema: StateSchema, n: int) -> Population: ...


class UniformInit(FrozenSpec):
    device: str = "cpu"

    def sample(self, schema: StateSchema, n: int) -> Population:
        return schema.uniform(n, self.device)


class CellSystem(FrozenSpec):
    dynamics: Tagged[Dynamics]
    observers: dict[str, Tagged[Observer]]
    init: Tagged[Init] = Field(default_factory=UniformInit)
    burn_in: int = Field(50, ge=0)
    signaling: Signaling = Field(default_factory=Signaling)

    def sample_cells(self, n: int, intervention: Intervention = CONTROL) -> Population:
        pop, _ = simulate(
            self.dynamics,
            self.init.sample(self.dynamics.state_schema, n),
            self.burn_in,
            intervention=intervention,
            signaling=self.signaling,
        )
        return pop

    def sample(
        self, n: int, intervention: Intervention = CONTROL, modality: str = "rna"
    ) -> torch.Tensor:
        return self.observers[modality].observe(self.sample_cells(n, intervention))

    def with_dynamics(self, dynamics: Dynamics) -> "CellSystem":
        return self.model_copy(update={"dynamics": dynamics})

    def description_length(self) -> float:
        return self.dynamics.description_length() + sum(
            o.description_length() for o in self.observers.values()
        )
