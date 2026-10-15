from typing import Protocol, Self, runtime_checkable

import torch
from pydantic import Field, PrivateAttr, model_validator

from histem.dynamics import Inputs
from histem.spec import FrozenSpec, Tagged
from histem.state import Population, Slot, StateSchema


@runtime_checkable
class Mechanism(Protocol):
    """One part of the shared rules: a step that rewrites some slots of a cell's
    state given the slots it reads."""

    @property
    def writes(self) -> tuple[Slot, ...]: ...

    @property
    def reads(self) -> tuple[str, ...]:
        """Names of every slot the step reads, its own included."""
        ...

    def step(self, pop: Population, inputs: Inputs) -> dict[str, torch.Tensor]:
        """New values of the written slots. Must not mutate `pop`."""
        ...


@runtime_checkable
class Emits(Protocol):
    """A mechanism that secretes signals and receives the routed ones back."""

    @property
    def signals(self) -> tuple[str, ...]: ...

    def emit_signals(self, pop: Population) -> torch.Tensor:
        """(n, len(signals))"""
        ...


class Composite(FrozenSpec):
    """Dynamics that run each mechanism in order every step, each seeing what the
    ones before it wrote. The state is the union of the written slots."""

    mechanisms: dict[str, Tagged[Mechanism]] = Field(min_length=1)

    _schema: StateSchema = PrivateAttr()

    @model_validator(mode="after")
    def _build(self) -> Self:
        slots: dict[str, Slot] = {}
        for name, m in self.mechanisms.items():
            for s in m.writes:
                if slots.setdefault(s.name, s) != s:
                    raise ValueError(f"{name}: slot {s.name!r} conflicts with another")
        for name, m in self.mechanisms.items():
            if unknown := sorted(set(m.reads) - slots.keys()):
                raise ValueError(f"{name} reads slots no mechanism writes: {unknown}")
        self._schema = StateSchema(slots=tuple(slots.values()))
        signals = [s for m in self._emitters() for s in m.signals]
        if len(set(signals)) != len(signals):
            raise ValueError("signal names must be unique across mechanisms")
        return self

    @property
    def state_schema(self) -> StateSchema:
        return self._schema

    @property
    def signals(self) -> tuple[str, ...]:
        return tuple(s for m in self._emitters() for s in m.signals)

    def parts(self) -> dict[str, object]:
        return dict(self.mechanisms)

    def _emitters(self) -> list[Emits]:
        return [m for m in self.mechanisms.values() if isinstance(m, Emits)]

    def step(self, pop: Population, inputs: Inputs) -> Population:
        offset = 0
        for m in self.mechanisms.values():
            k = len(m.signals) if isinstance(m, Emits) else 0
            received = inputs.signals[:, offset : offset + k]
            offset += k
            written = m.step(pop, Inputs(received, inputs.intervention, inputs.context))
            pop = Population(self._schema, pop.values | written)
        return pop

    def emit_signals(self, pop: Population) -> torch.Tensor:
        parts = [m.emit_signals(pop) for m in self._emitters()]
        return (
            torch.cat(parts, dim=1)
            if parts
            else torch.zeros((pop.n, 0), device=pop.device)
        )
