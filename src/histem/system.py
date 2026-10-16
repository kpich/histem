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
class HasGeneRows(Protocol):
    def gene_rows(self) -> dict[str, tuple[str, ...]]: ...


@runtime_checkable
class Learnable(Protocol):
    def named_parameters(self) -> Iterator[tuple[str, torch.Tensor]]: ...


class CellSystem(FrozenSpec):
    dynamics: Tagged[Dynamics]
    observers: dict[str, Tagged[Observer]]
    init: Tagged[Init] = Field(default_factory=UniformInit)
    # the one owner of the gene vectors that dynamics and observers share
    genes: GeneTable | None = None
    # dynamics steps from the initial state to the observed cells
    steps: int = Field(50, ge=0)
    # under autograd, backprop through only the last `grad_steps` steps
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

    def sample_cells(
        self, n: int, context: str, intervention: Intervention = CONTROL
    ) -> Population:
        pop = self.init.sample(self.dynamics.state_schema, n, self.device)
        tracked = self.steps
        if self.grad_steps is not None and torch.is_grad_enabled():
            tracked = min(self.grad_steps, self.steps)
            with torch.no_grad():
                pop = simulate(
                    self.dynamics,
                    pop,
                    self.steps - tracked,
                    context=context,
                    intervention=intervention,
                    signaling=self.signaling,
                )
        return simulate(
            self.dynamics,
            pop,
            tracked,
            context=context,
            intervention=intervention,
            signaling=self.signaling,
        )

    def sample(
        self,
        n: int,
        context: str,
        intervention: Intervention = CONTROL,
        modality: str = "rna",
        *,
        reparam: bool = False,
    ) -> torch.Tensor:
        observer = self.observers[modality]
        cells = self.sample_cells(n, context, intervention)
        if reparam:
            return observer.rsample(cells, context)
        return observer.observe(cells, context)

    def named_parameters(self) -> Iterator[tuple[str, torch.Tensor]]:
        for prefix, part in self._parts().items():
            if isinstance(part, Learnable):
                for name, p in part.named_parameters():
                    yield f"{prefix}/{name}", p

    def parameters(self) -> Iterator[torch.Tensor]:
        for _, p in self.named_parameters():
            yield p

    def _gene_rows(self) -> dict[str, tuple[str, ...]]:
        out: dict[str, tuple[str, ...]] = {}
        for prefix, part in self._parts().items():
            if isinstance(part, HasGeneRows):
                out |= {f"{prefix}/{k}": v for k, v in part.gene_rows().items()}
        return out

    def load_weights(self, src: "CellSystem") -> list[str]:
        """Copy `src`'s parameters into this system by name, so parts can be
        swapped or added. Parameters with gene rows copy the genes both have, so a
        gene table can grow; genes `src` lacks get zero table vectors, which leave
        the parts trained without them unchanged. Returns the names of parameters
        `src` doesn't have, which keep their initial values."""
        theirs = dict(src.named_parameters())
        their_rows, our_rows = src._gene_rows(), self._gene_rows()
        fresh = []
        with torch.no_grad():
            for name, p in self.named_parameters():
                if name not in theirs:
                    fresh.append(name)
                    continue
                q = theirs[name].to(p.device)
                if name in our_rows:
                    where = {g: i for i, g in enumerate(their_rows[name])}
                    pairs = [
                        (i, where[g])
                        for i, g in enumerate(our_rows[name])
                        if g in where
                    ]
                    if pairs:
                        dst, srcs = zip(*pairs, strict=True)
                        p[list(dst)] = q[list(srcs)]
                elif p.shape != q.shape:
                    raise ValueError(
                        f"{name}: shape {tuple(q.shape)} != {tuple(p.shape)}"
                    )
                else:
                    p.copy_(q)
            if self.genes is not None and src.genes is not None:
                table = self.genes.weights["table"]
                known = set(src.genes.genes)
                new = [i for i, g in enumerate(self.genes.genes) if g not in known]
                table[new] = 0
        return fresh

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
