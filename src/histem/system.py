from collections.abc import Callable

import anndata as ad
import numpy as np
from pydantic import Field

from histem.dynamics import CONTROL, Dynamics, Intervention
from histem.observers import Observer
from histem.simulator import Signaling, simulate
from histem.spec import FrozenSpec
from histem.state import Population

InitPrior = Callable[[int, np.random.Generator], Population]


class CellSystem(FrozenSpec):
    dynamics: Dynamics
    observers: dict[str, Observer]
    init: InitPrior
    burn_in: int = Field(50, ge=0)
    signaling: Signaling = Field(default_factory=Signaling)

    def sample_cells(
        self, n: int, rng: np.random.Generator, intervention: Intervention = CONTROL
    ) -> Population:
        pop, _ = simulate(
            self.dynamics,
            self.init(n, rng),
            self.burn_in,
            rng,
            intervention=intervention,
            signaling=self.signaling,
        )
        return pop

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
        intervention: Intervention = CONTROL,
        modality: str = "rna",
    ) -> ad.AnnData:
        adata = self.observers[modality].observe(
            self.sample_cells(n, rng, intervention), rng
        )
        adata.obs["condition"] = intervention.name
        return adata

    def with_dynamics(self, dynamics: Dynamics) -> "CellSystem":
        return self.model_copy(update={"dynamics": dynamics})

    def description_length(self) -> float:
        return self.dynamics.description_length() + sum(
            o.description_length() for o in self.observers.values()
        )
