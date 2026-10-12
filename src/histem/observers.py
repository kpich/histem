import math
from collections.abc import Iterator
from typing import Protocol, Self, runtime_checkable

import torch
from pydantic import Field, PrivateAttr, model_validator
from torch import nn

from histem.genes import GeneTable
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


def state_features(pop: Population, variables: tuple[str, ...]) -> torch.Tensor:
    """(n, k) features, each divided by its slot's levels - 1."""
    cols = []
    for v in variables:
        slot = pop.schema.slot(pop.schema.locate(v)[0])
        cols.append(pop.get(v) / (slot.levels - 1))
    return torch.stack(cols, dim=1)


class _Decoder(nn.Module):
    def __init__(self, n_drivers: int, n_genes: int, width: int, gene_dim: int):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(n_drivers, width), nn.SiLU())
        # a gene's output weights are a projection of its table vector
        self.out = nn.Linear(gene_dim, width, bias=False)
        nn.init.zeros_(self.out.weight)
        self.bias = nn.Parameter(torch.zeros(n_genes))
        self.log_dispersion = nn.Parameter(torch.full((n_genes,), math.log(5.0)))

    def forward(self, features: torch.Tensor, genes: torch.Tensor) -> torch.Tensor:
        """`genes` (n_genes, gene_dim) are the observed genes' table vectors."""
        out: torch.Tensor = self.bias + self.hidden(features) @ self.out(genes).T
        return out


class DecoderObserver(FrozenSpec):
    """log mean_g = log(size) + bias_g + MLP(features(state)) . W e_g, where e_g is
    gene g's vector in the bound `GeneTable`, with a learned per-gene NB
    dispersion. W starts at zero, so set `bias` to the data's log mean counts to
    start at the right scale."""

    genes: tuple[str, ...] = Field(min_length=1)
    drivers: tuple[str, ...] = Field(min_length=1)
    width: int = Field(128, gt=0)
    gene_dim: int = Field(128, gt=0)  # the bound GeneTable's dim
    size_sd: float = Field(0.3, ge=0)  # sd of log library size
    modality: str = "rna"
    # empty means freshly initialized; afterwards it holds the live parameters
    weights: dict[str, Tensor] = Field(default_factory=dict)

    _net: _Decoder = PrivateAttr()
    _genes: GeneTable | None = PrivateAttr(None)

    @model_validator(mode="after")
    def _build(self) -> Self:
        if len(set(self.genes)) != len(self.genes):
            raise ValueError("duplicate gene names")
        self._net = _Decoder(
            len(self.drivers), len(self.genes), self.width, self.gene_dim
        )
        if self.weights:
            self._net.load_state_dict(self.weights, assign=True)
        self.weights.clear()
        self.weights.update(self._net.named_parameters())
        return self

    @property
    def features(self) -> tuple[str, ...]:
        return self.genes

    def named_parameters(self) -> Iterator[tuple[str, torch.Tensor]]:
        return iter(self.weights.items())

    def bind_genes(self, table: GeneTable) -> None:
        if table.dim != self.gene_dim:
            raise ValueError(f"gene table dim {table.dim} != gene_dim {self.gene_dim}")
        self._genes = table

    def set_bias(self, bias: torch.Tensor) -> None:
        with torch.no_grad():
            self._net.bias.copy_(bias)

    def log_mean(self, pop: Population) -> torch.Tensor:
        log_size = self.size_sd * torch.randn((pop.n, 1), device=pop.device)
        if self._genes is None:
            raise RuntimeError("no gene table bound")
        features = state_features(pop, self.drivers)
        out: torch.Tensor = log_size + self._net(
            features, self._genes.embed(self.genes)
        )
        return out

    def observe(self, pop: Population) -> torch.Tensor:
        return _nb_sample(self.log_mean(pop), self._net.log_dispersion.exp())

    def rsample(self, pop: Population) -> torch.Tensor:
        return _nb_rsample(self.log_mean(pop), self._net.log_dispersion.exp())


def _nb_sample(log_mean: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    # torch's NB counts failures before total_count successes, so this
    # parameterization has mean exp(log_mean) and inverse dispersion theta
    nb = torch.distributions.NegativeBinomial(theta, logits=log_mean - theta.log())
    counts: torch.Tensor = nb.sample()
    return counts


def _nb_rsample(log_mean: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    # NB = Poisson(Gamma): the gamma rate is reparameterized, the Poisson
    # counts are exact with a straight-through gradient to the rate
    rate = torch.distributions.Gamma(theta, theta / log_mean.exp()).rsample()
    return rate + (torch.poisson(rate.detach()) - rate).detach()
