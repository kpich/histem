from dataclasses import dataclass, field
from typing import Any, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from histem.spec import FrozenSpec

SlotKind = Literal["discrete", "continuous"]


class Slot(FrozenSpec):
    name: str
    variables: tuple[str, ...] = Field(min_length=1)
    kind: SlotKind = "discrete"
    levels: int = 2  # discrete values are 0..levels-1
    observed: bool = False

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.kind == "discrete" and not 2 <= self.levels <= 127:  # stored as int8
            raise ValueError(f"slot {self.name!r}: levels must be in [2, 127]")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError(f"slot {self.name!r}: duplicate variable names")
        return self

    @property
    def dim(self) -> int:
        return len(self.variables)

    @property
    def dtype(self) -> type[np.generic]:
        return np.int8 if self.kind == "discrete" else np.float32


class StateSchema(FrozenSpec):
    slots: tuple[Slot, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        names = [v for s in self.slots for v in s.variables]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(
                f"variable names must be unique across slots: {sorted(dupes)}"
            )
        slot_names = [s.name for s in self.slots]
        if len(set(slot_names)) != len(slot_names):
            raise ValueError("slot names must be unique")
        return self

    def slot(self, name: str) -> Slot:
        for s in self.slots:
            if s.name == name:
                return s
        raise KeyError(name)

    def locate(self, variable: str) -> tuple[str, int]:
        """Variable name -> (slot name, column index)."""
        for s in self.slots:
            if variable in s.variables:
                return s.name, s.variables.index(variable)
        raise KeyError(variable)

    @property
    def variables(self) -> list[str]:
        return [v for s in self.slots for v in s.variables]

    def empty(self, n: int) -> "Population":
        return Population(
            self, {s.name: np.zeros((n, s.dim), s.dtype) for s in self.slots}
        )

    def uniform(self, n: int, rng: np.random.Generator) -> "Population":
        """Discrete slots uniform over levels, continuous slots standard normal."""
        values = {}
        for s in self.slots:
            if s.kind == "discrete":
                values[s.name] = rng.integers(0, s.levels, (n, s.dim)).astype(s.dtype)
            else:
                values[s.name] = rng.standard_normal((n, s.dim)).astype(s.dtype)
        return Population(self, values)


@dataclass
class Population:
    """`values[slot]` has shape (n_cells, slot.dim)."""

    schema: StateSchema
    values: dict[str, np.ndarray]
    positions: np.ndarray | None = None  # (n, d)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(next(iter(self.values.values())).shape[0])

    def get(self, variable: str) -> np.ndarray:
        slot, i = self.schema.locate(variable)
        return self.values[slot][:, i]

    def variable_view(self) -> dict[str, np.ndarray]:
        """Flat {variable: (n,) array} view, without copying."""
        return {
            v: self.values[s.name][:, i]
            for s in self.schema.slots
            for i, v in enumerate(s.variables)
        }

    def copy(self) -> "Population":
        pos = None if self.positions is None else self.positions.copy()
        return Population(
            self.schema,
            {k: v.copy() for k, v in self.values.items()},
            pos,
            dict(self.meta),
        )

    def subset(self, idx: np.ndarray | slice) -> "Population":
        pos = None if self.positions is None else self.positions[idx]
        return Population(
            self.schema,
            {k: v[idx] for k, v in self.values.items()},
            pos,
            dict(self.meta),
        )
