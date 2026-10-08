"""A WorldModel is everything needed to produce data: shared rules, a prior over
initial cell states, how signals are routed, and an Observer per modality."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import anndata as ad
import numpy as np

from histem.dynamics import CONTROL, Dynamics, Intervention
from histem.observers import Observer
from histem.simulator import Signaling, simulate
from histem.state import Population

InitPrior = Callable[[int, np.random.Generator], Population]


@dataclass
class WorldModel:
    dynamics: Dynamics
    observers: dict[str, Observer]
    init: InitPrior
    burn_in: int = (
        50  # steps run before observing; snapshots are treated as near-stationary
    )
    signaling: Signaling = field(default_factory=Signaling)

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

    def description_length(self) -> float:
        return self.dynamics.description_length() + sum(
            o.description_length() for o in self.observers.values()
        )
