from collections.abc import Callable

import torch
from pydantic import Field

from histem.dynamics import CONTROL, Dynamics, Intervention
from histem.observers import Observer
from histem.simulator import Signaling, simulate
from histem.spec import FrozenSpec
from histem.state import Population

InitPrior = Callable[[int], Population]


class CellSystem(FrozenSpec):
    dynamics: Dynamics
    observers: dict[str, Observer]
    init: InitPrior
    burn_in: int = Field(50, ge=0)
    signaling: Signaling = Field(default_factory=Signaling)

    def sample_cells(self, n: int, intervention: Intervention = CONTROL) -> Population:
        pop, _ = simulate(
            self.dynamics,
            self.init(n),
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
