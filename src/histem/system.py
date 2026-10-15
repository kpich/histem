from collections.abc import Iterator
from typing import Protocol, Self, runtime_checkable

import torch
from pydantic import Field, model_validator

from histem.dynamics import CONTROL, Dynamics, Intervention
from histem.genes import GeneTable, UsesGenes
from histem.observers import Observer
from histem.simulator import Signaling, simulate
from histem.spec import FrozenSpec, Tagged
from histem.state import Population, StateSchema


@runtime_checkable
class Init(Protocol):
    def sample(self, schema: StateSchema, n: int, device: str) -> Population: ...


class UniformInit(FrozenSpec):
    def sample(self, schema: StateSchema, n: int, device: str) -> Population:
        return schema.uniform(n, device)


@runtime_checkable
class HasParts(Protocol):
    def parts(self) -> dict[str, object]: ...


@runtime_checkable
class Learnable(Protocol):
    def named_parameters(self) -> Iterator[tuple[str, torch.Tensor]]: ...


class CellSystem(FrozenSpec):
    dynamics: Tagged[Dynamics]
    observers: dict[str, Tagged[Observer]]
    init: Tagged[Init] = Field(default_factory=UniformInit)
    # the one owner of the gene vectors that dynamics and observers share
    genes: GeneTable | None = None
    burn_in: int = Field(50, ge=0)
    # under autograd, backprop through only the last `grad_steps` of burn-in
    grad_steps: int | None = Field(None, ge=0)
    signaling: Signaling = Field(default_factory=Signaling)
    device: str = "cpu"

    @model_validator(mode="after")
    def _bind_genes(self) -> Self:
        for part in self._parts().values():
            if isinstance(part, UsesGenes):
                if self.genes is None:
                    raise ValueError(f"{type(part).__name__} needs a gene table")
                part.bind_genes(self.genes)
        return self

    def _parts(self) -> dict[str, object]:
        parts = {"dynamics": self.dynamics, "init": self.init, "genes": self.genes}
        parts |= {f"observers/{k}": v for k, v in self.observers.items()}
        return _flatten(parts)

    def sample_cells(self, n: int, intervention: Intervention = CONTROL) -> Population:
        pop = self.init.sample(self.dynamics.state_schema, n, self.device)
        tracked = self.burn_in
        if self.grad_steps is not None and torch.is_grad_enabled():
            tracked = min(self.grad_steps, self.burn_in)
            with torch.no_grad():
                pop = simulate(
                    self.dynamics,
                    pop,
                    self.burn_in - tracked,
                    intervention=intervention,
                    signaling=self.signaling,
                )
        return simulate(
            self.dynamics,
            pop,
            tracked,
            intervention=intervention,
            signaling=self.signaling,
        )

    def sample(
        self,
        n: int,
        intervention: Intervention = CONTROL,
        modality: str = "rna",
        *,
        reparam: bool = False,
    ) -> torch.Tensor:
        observer = self.observers[modality]
        cells = self.sample_cells(n, intervention)
        return observer.rsample(cells) if reparam else observer.observe(cells)

    def named_parameters(self) -> Iterator[tuple[str, torch.Tensor]]:
        for prefix, part in self._parts().items():
            if isinstance(part, Learnable):
                for name, p in part.named_parameters():
                    yield f"{prefix}/{name}", p

    def parameters(self) -> Iterator[torch.Tensor]:
        for _, p in self.named_parameters():
            yield p

    def to(self, device: str) -> "CellSystem":
        """A copy with every tensor on `device`; learnable parameters are new."""
        tensors: dict[str, torch.Tensor] = {}
        config = self.model_dump(context={"tensors": tensors})
        config["device"] = device
        moved = {k: v.to(device) for k, v in tensors.items()}
        return self.model_validate(config, context={"tensors": moved})


def _flatten(parts: dict[str, object]) -> dict[str, object]:
    """Parts and, recursively, their sub-parts as `<part>/<sub-part>`."""
    out = dict(parts)
    for prefix, part in parts.items():
        if isinstance(part, HasParts):
            sub = _flatten(part.parts())
            out |= {f"{prefix}/{k}": v for k, v in sub.items()}
    return out
