import math
from typing import Protocol, Self, runtime_checkable

import torch
from pydantic import Field, model_validator

from histem.spec import FrozenSpec, Tensor
from histem.state import Population


@runtime_checkable
class Observer(Protocol):
    @property
    def modality(self) -> str: ...

    @property
    def features(self) -> tuple[str, ...]:
        """Names of the observed columns, e.g. genes."""
        ...

    def observe(self, pop: Population) -> torch.Tensor:
        """(n_cells, n_features)"""
        ...

    def rsample(self, pop: Population) -> torch.Tensor:
        """A reparameterized stand-in for `observe`, for gradient training."""
        ...

    def description_length(self) -> float: ...


def state_features(pop: Population, variables: tuple[str, ...]) -> torch.Tensor:
    """(n, k) features: discrete scaled to [0, 1], continuous passed through."""
    cols = []
    for v in variables:
        slot = pop.schema.slot(pop.schema.locate(v)[0])
        x = pop.get(v).float()
        cols.append(x / (slot.levels - 1) if slot.kind == "discrete" else x)
    return torch.stack(cols, dim=1)


class NBCountObserver(FrozenSpec):
    """log mean_g = log(size) + bias_g + weights_g . features(state)"""

    genes: tuple[str, ...] = Field(min_length=1)
    drivers: tuple[str, ...] = Field(min_length=1)
    weights: Tensor  # (n_genes, n_drivers)
    bias: Tensor  # (n_genes,)
    dispersion: float = Field(5.0, gt=0)  # NB theta
    size_sd: float = Field(0.3, ge=0)  # sd of log library size
    modality: str = "rna"

    @model_validator(mode="after")
    def _check(self) -> Self:
        expected = (len(self.genes), len(self.drivers))
        if tuple(self.weights.shape) != expected:
            raise ValueError(f"weights shape {tuple(self.weights.shape)} != {expected}")
        if tuple(self.bias.shape) != (len(self.genes),):
            raise ValueError(
                f"bias shape {tuple(self.bias.shape)} != ({len(self.genes)},)"
            )
        if len(set(self.genes)) != len(self.genes):
            raise ValueError("duplicate gene names")
        return self

    @property
    def features(self) -> tuple[str, ...]:
        return self.genes

    def log_mean(self, pop: Population) -> torch.Tensor:
        log_size = self.size_sd * torch.randn((pop.n, 1), device=pop.device)
        w, b = self.weights.to(pop.device), self.bias.to(pop.device)
        return log_size + b + state_features(pop, self.drivers) @ w.T

    def observe(self, pop: Population) -> torch.Tensor:
        # torch's NB counts failures before total_count successes, so this
        # parameterization has mean exp(log_mean) and inverse dispersion theta
        logits = self.log_mean(pop) - math.log(self.dispersion)
        nb = torch.distributions.NegativeBinomial(self.dispersion, logits=logits)
        counts: torch.Tensor = nb.sample()
        return counts

    def rsample(self, pop: Population) -> torch.Tensor:
        # NB = Poisson(Gamma): the gamma rate is reparameterized, the Poisson step
        # is replaced by a moment-matched normal
        mean = self.log_mean(pop).exp()
        theta = torch.tensor(self.dispersion, device=mean.device)
        rate = torch.distributions.Gamma(theta, theta / mean).rsample()
        return (rate + rate.sqrt() * torch.randn_like(rate)).clamp(min=0)

    def description_length(self) -> float:
        return float(torch.count_nonzero(self.weights) + self.bias.numel()) * 8.0
