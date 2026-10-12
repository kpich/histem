from collections.abc import Iterator
from typing import Protocol, Self, runtime_checkable

import torch
from pydantic import Field, model_validator

from histem.spec import FrozenSpec, Tensor


class GeneTable(FrozenSpec):
    """One learned vector per gene, shared by every part of a system that refers to
    genes, e.g. as a perturbation target and as an expression readout."""

    genes: tuple[str, ...] = Field(min_length=1)
    dim: int = Field(128, gt=0)
    # empty means freshly initialized; afterwards it holds the live parameters
    weights: dict[str, Tensor] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _build(self) -> Self:
        if len(set(self.genes)) != len(self.genes):
            raise ValueError("duplicate gene names")
        if not self.weights:
            table = torch.randn(len(self.genes), self.dim) / self.dim**0.5
            self.weights["table"] = torch.nn.Parameter(table)
        elif not isinstance(self.weights["table"], torch.nn.Parameter):
            self.weights["table"] = torch.nn.Parameter(self.weights["table"])
        return self

    def named_parameters(self) -> Iterator[tuple[str, torch.Tensor]]:
        return iter(self.weights.items())

    def index(self, genes: tuple[str, ...], device: torch.device) -> torch.Tensor:
        where = {g: i for i, g in enumerate(self.genes)}
        if unknown := sorted(set(genes) - where.keys()):
            raise KeyError(f"unknown genes {unknown}")
        return torch.tensor([where[g] for g in genes], dtype=torch.long, device=device)

    def embed(self, genes: tuple[str, ...]) -> torch.Tensor:
        """(len(genes), dim)"""
        table = self.weights["table"]
        return table[self.index(genes, table.device)]


@runtime_checkable
class UsesGenes(Protocol):
    def bind_genes(self, table: GeneTable) -> None:
        """Called by the owning system; the part reads the table but doesn't save
        it."""
        ...
