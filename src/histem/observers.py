"""Observation models: hidden cell state -> what an assay measures.

Each modality (scRNA counts, ATAC, spatial, Hi-C, ...) is an Observer. Adding a data
type means adding an Observer; the Dynamics don't change.
"""

from typing import Protocol, Self, runtime_checkable

import anndata as ad
import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from histem.spec import FrozenSpec
from histem.state import Population


@runtime_checkable
class Observer(Protocol):
    @property
    def modality(self) -> str: ...

    def observe(self, pop: Population, rng: np.random.Generator) -> ad.AnnData: ...

    def description_length(self) -> float: ...


def state_features(pop: Population, variables: tuple[str, ...]) -> np.ndarray:
    """(n, k) features: discrete scaled to [0, 1], continuous passed through."""
    cols = []
    for v in variables:
        slot = pop.schema.slot(pop.schema.locate(v)[0])
        x = pop.get(v).astype(np.float32)
        cols.append(x / (slot.levels - 1) if slot.kind == "discrete" else x)
    return np.stack(cols, axis=1)


class NBCountObserver(FrozenSpec):
    """scRNA-like counts.

    log mean_g = log(size) + bias_g + weights_g . features(state)
    """

    genes: tuple[str, ...] = Field(min_length=1)
    drivers: tuple[str, ...] = Field(min_length=1)  # state variables counts depend on
    weights: np.ndarray  # (n_genes, n_drivers)
    bias: np.ndarray  # (n_genes,)
    dispersion: float = Field(5.0, gt=0)  # NB theta; larger = closer to Poisson
    size_sd: float = Field(0.3, ge=0)  # lognormal per-cell library-size noise
    modality: str = "rna"

    @model_validator(mode="after")
    def _check(self) -> Self:
        expected = (len(self.genes), len(self.drivers))
        if self.weights.shape != expected:
            raise ValueError(f"weights shape {self.weights.shape} != {expected}")
        if self.bias.shape != (len(self.genes),):
            raise ValueError(f"bias shape {self.bias.shape} != ({len(self.genes)},)")
        if len(set(self.genes)) != len(self.genes):
            raise ValueError("duplicate gene names")
        return self

    def mean(self, pop: Population, rng: np.random.Generator) -> np.ndarray:
        size = np.exp(rng.normal(0.0, self.size_sd, (pop.n, 1)))
        log_mu = self.bias + state_features(pop, self.drivers) @ self.weights.T
        mu: np.ndarray = size * np.exp(log_mu)
        return mu

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
