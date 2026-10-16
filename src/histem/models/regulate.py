from collections.abc import Iterator
from typing import Self

import torch
from pydantic import Field, PrivateAttr, model_validator
from torch import nn

from histem.dynamics import Inputs
from histem.genes import GeneTable
from histem.spec import FrozenSpec, Tensor
from histem.state import Population, Slot


class _Net(nn.Module):
    def __init__(self, gene_dim: int, rank: int):
        super().__init__()
        # no biases, so a gene with a zero table vector neither drives nor is driven
        self.source = nn.Linear(gene_dim, rank, bias=False)
        self.sink = nn.Linear(gene_dim, rank, bias=False)
        nn.init.zeros_(self.sink.weight)
        self.rate = nn.Parameter(torch.zeros(()))  # sigmoid 0.5

    def forward(self, a: torch.Tensor, table: torch.Tensor, cap: float) -> torch.Tensor:
        h = (a @ self.source(table)) @ self.sink(table).T
        rate = torch.sigmoid(self.rate)
        out: torch.Tensor = (1 - rate) * a + rate * cap * torch.tanh(h / cap)
        return out


class Regulate(FrozenSpec):
    """A mechanism that relaxes each gene's activity in `slot` toward a saturating
    function of every gene's activity, through a rank-`rank` coupling built from the
    gene table: gene j drives gene g by source(j) · sink(g). Run it before `Perturb`
    so the targets stay held."""

    slot: Slot
    gene_dim: int = Field(128, gt=0)  # the bound GeneTable's dim
    rank: int = Field(16, gt=0)
    cap: float = Field(3.0, gt=0)  # bound on a driven gene's log fold change
    # empty means freshly initialized; afterwards it holds the live parameters
    weights: dict[str, Tensor] = Field(default_factory=dict)

    _net: _Net = PrivateAttr()
    _genes: GeneTable | None = PrivateAttr(None)

    @model_validator(mode="after")
    def _build(self) -> Self:
        if self.slot.levels is not None:
            raise ValueError("the activity slot must be continuous")
        self._net = _Net(self.gene_dim, self.rank)
        if self.weights:
            self._net.load_state_dict(self.weights, assign=True)
        self.weights.clear()
        self.weights.update(self._net.named_parameters())
        return self

    @property
    def writes(self) -> tuple[Slot, ...]:
        return (self.slot,)

    @property
    def reads(self) -> tuple[str, ...]:
        return (self.slot.name,)

    def named_parameters(self) -> Iterator[tuple[str, torch.Tensor]]:
        return iter(self.weights.items())

    def bind_genes(self, table: GeneTable) -> None:
        if table.dim != self.gene_dim:
            raise ValueError(f"gene table dim {table.dim} != gene_dim {self.gene_dim}")
        self._genes = table

    def step(self, pop: Population, inputs: Inputs) -> dict[str, torch.Tensor]:
        if self._genes is None:
            raise RuntimeError("no gene table bound")
        table = self._genes.embed(self.slot.variables)
        a = pop.values[self.slot.name]
        return {self.slot.name: self._net(a, table, self.cap)}
