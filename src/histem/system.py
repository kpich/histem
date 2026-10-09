from collections.abc import Iterator
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
    def sample(self, schema: StateSchema, n: int, device: str) -> Population: ...


class UniformInit(FrozenSpec):
    def sample(self, schema: StateSchema, n: int, device: str) -> Population:
        return schema.uniform(n, device)


@runtime_checkable
class Learnable(Protocol):
    def parameters(self) -> Iterator[torch.Tensor]: ...


class CellSystem(FrozenSpec):
    dynamics: Tagged[Dynamics]
    observers: dict[str, Tagged[Observer]]
    init: Tagged[Init] = Field(default_factory=UniformInit)
    burn_in: int = Field(50, ge=0)
    signaling: Signaling = Field(default_factory=Signaling)
    device: str = "cpu"

    def sample_cells(self, n: int, intervention: Intervention = CONTROL) -> Population:
        pop, _ = simulate(
            self.dynamics,
            self.init.sample(self.dynamics.state_schema, n, self.device),
            self.burn_in,
            intervention=intervention,
            signaling=self.signaling,
        )
        return pop

    def sample(
        self,
        n: int,
        intervention: Intervention = CONTROL,
        modality: str = "rna",
        *,
        reparam: bool = False,
    ) -> torch.Tensor:
        observer = self.observers[modality]
        cells = self.sample_cells(n, intervention)
        return observer.rsample(cells) if reparam else observer.observe(cells)

    def with_dynamics(self, dynamics: Dynamics) -> "CellSystem":
        return self.model_copy(update={"dynamics": dynamics})

    def parameters(self) -> Iterator[torch.Tensor]:
        for part in [self.dynamics, self.init, *self.observers.values()]:
            if isinstance(part, Learnable):
                yield from part.parameters()

    def to(self, device: str) -> "CellSystem":
        """A copy with every tensor on `device`; learnable parameters are new."""
        tensors: dict[str, torch.Tensor] = {}
        config = self.model_dump(context={"tensors": tensors})
        config["device"] = device
        moved = {k: v.to(device) for k, v in tensors.items()}
        return self.model_validate(config, context={"tensors": moved})

    def description_length(self) -> float:
        return self.dynamics.description_length() + sum(
            o.description_length() for o in self.observers.values()
        )
