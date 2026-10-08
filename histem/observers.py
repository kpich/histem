"""Observation models: hidden cell state -> what an assay measures.

Each modality (scRNA counts, ATAC, spatial, Hi-C, ...) is an Observer. Adding a data
type means adding an Observer; the Dynamics don't change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import anndata as ad
import numpy as np
import pandas as pd

from histem.state import Population


class Observer(Protocol):
    modality: str

    def observe(self, pop: Population, rng: np.random.Generator) -> ad.AnnData: ...

    def description_length(self) -> float: ...


def state_features(pop: Population, variables: tuple[str, ...]) -> np.ndarray:
    """(n, k) features: discrete variables scaled to [0, 1], continuous ones passed through."""
    cols = []
    for v in variables:
        slot = pop.schema.slot(pop.schema.locate(v)[0])
        x = pop.get(v).astype(np.float32)
        cols.append(x / (slot.levels - 1) if slot.kind == "discrete" else x)
    return np.stack(cols, axis=1)


@dataclass
class NBCountObserver:
    """scRNA-like counts. log mean_g = log(size) + bias_g + weights_g . features(state)."""

    genes: tuple[str, ...]
    drivers: tuple[str, ...]  # state variables the counts depend on
    weights: np.ndarray  # (n_genes, n_drivers)
    bias: np.ndarray  # (n_genes,)
    dispersion: float = 5.0  # NB theta; larger = closer to Poisson
    size_sd: float = 0.3  # lognormal per-cell library-size noise
    modality: str = "rna"

    def mean(self, pop: Population, rng: np.random.Generator) -> np.ndarray:
        size = np.exp(rng.normal(0.0, self.size_sd, (pop.n, 1)))
        return size * np.exp(self.bias + state_features(pop, self.drivers) @ self.weights.T)

    def observe(self, pop: Population, rng: np.random.Generator) -> ad.AnnData:
        mu = self.mean(pop, rng)
        p = self.dispersion / (self.dispersion + mu)
        counts = rng.negative_binomial(self.dispersion, p).astype(np.float32)
        return ad.AnnData(
            X=counts,
            obs=pd.DataFrame(index=[f"cell{i}" for i in range(pop.n)]),
            var=pd.DataFrame(index=list(self.genes)),
        )

    def description_length(self) -> float:
        return float(np.count_nonzero(self.weights) + self.bias.size) * 8.0
