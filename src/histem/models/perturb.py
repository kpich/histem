from collections.abc import Iterator
from typing import Self

import torch
from pydantic import Field, PrivateAttr, model_validator
from torch import nn

from histem.dynamics import KINDS, Inputs, Kind
from histem.genes import GeneTable
from histem.spec import FrozenSpec, Tensor
from histem.state import Population, Slot

# activity is log fold change vs control, so knockout is strongly negative
_INIT = {"activate": 2.0, "repress": -2.0, "knockout": -4.0}


class _Net(nn.Module):
    def __init__(self, gene_dim: int):
        super().__init__()
        self.base = nn.Parameter(torch.tensor([_INIT[k] for k in KINDS]))
        self.gene = nn.Linear(gene_dim, len(KINDS))
        nn.init.zeros_(self.gene.weight)
        nn.init.zeros_(self.gene.bias)

    def forward(self, genes: torch.Tensor, kind: int) -> torch.Tensor:
        out: torch.Tensor = self.base[kind] + self.gene(genes)[:, kind]
        return out


class Perturb(FrozenSpec):
    """A mechanism that holds each perturbed gene's activity in `slot` at a learned
    magnitude: one per kind plus a per-gene term from the gene table."""

    slot: Slot
    gene_dim: int = Field(128, gt=0)  # the bound GeneTable's dim
    # empty means freshly initialized; afterwards it holds the live parameters
    weights: dict[str, Tensor] = Field(default_factory=dict)

    _net: _Net = PrivateAttr()
    _genes: GeneTable | None = PrivateAttr(None)

    @model_validator(mode="after")
    def _build(self) -> Self:
        if self.slot.levels is not None:
            raise ValueError("the activity slot must be continuous")
        self._net = _Net(self.gene_dim)
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

    def magnitude(self, genes: tuple[str, ...], kind: Kind) -> torch.Tensor:
        if self._genes is None:
            raise RuntimeError("no gene table bound")
        out: torch.Tensor = self._net(self._genes.embed(genes), KINDS.index(kind))
        return out

    def step(self, pop: Population, inputs: Inputs) -> dict[str, torch.Tensor]:
        activity = pop.values[self.slot.name]
        iv = inputs.intervention
        if not iv.genes:
            return {self.slot.name: activity}
        assert iv.kind is not None
        where = {g: i for i, g in enumerate(self.slot.variables)}
        if unknown := sorted(set(iv.genes) - where.keys()):
            raise KeyError(f"unknown genes {unknown}")
        idx = torch.tensor([where[g] for g in iv.genes], device=activity.device)
        out = activity.clone()
        out[:, idx] = self.magnitude(iv.genes, iv.kind)
        return {self.slot.name: out}
