from dataclasses import dataclass, field
from typing import Literal, Protocol, Self, get_args, runtime_checkable

import torch
from pydantic import model_validator

from histem.spec import FrozenSpec
from histem.state import Population, StateSchema

Kind = Literal["activate", "repress", "knockout"]
KINDS: tuple[Kind, ...] = get_args(Kind)


class Intervention(FrozenSpec):
    """`genes` are perturbation targets, which a dynamics model reads as an input;
    `kind` is what is done to them."""

    name: str = "control"
    genes: tuple[str, ...] = ()
    kind: Kind | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if bool(self.genes) != (self.kind is not None):
            raise ValueError(f"{self.name}: kind is set iff there are genes")
        return self


CONTROL = Intervention()


@dataclass
class Inputs:
    signals: torch.Tensor  # (n, n_signals) received
    intervention: Intervention = field(default=CONTROL)
    context: str | None = None  # e.g. the dataset's cell type and assay


@runtime_checkable
class Dynamics(Protocol):
    """Shared rules: a stochastic transition kernel over per-cell state."""

    @property
    def state_schema(self) -> StateSchema: ...

    def step(self, pop: Population, inputs: Inputs) -> Population:
        """Must not mutate `pop`."""
        ...

    def emit_signals(self, pop: Population) -> torch.Tensor:
        """(n, n_signals) secreted amounts; routing is the simulator's job."""
        ...
