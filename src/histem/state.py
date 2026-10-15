from dataclasses import dataclass
from typing import Self

import torch
from pydantic import Field, model_validator

from histem.spec import FrozenSpec


class Slot(FrozenSpec):
    name: str
    variables: tuple[str, ...] = Field(min_length=1)
    # values live on the grid 0..levels-1, or anywhere if None (continuous)
    levels: int | None = Field(2, ge=2)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError(f"slot {self.name!r}: duplicate variable names")
        return self

    @property
    def dim(self) -> int:
        return len(self.variables)


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

    def empty(self, n: int, device: torch.device | str = "cpu") -> "Population":
        return Population(
            self,
            {s.name: torch.zeros((n, s.dim), device=device) for s in self.slots},
        )

    def uniform(self, n: int, device: torch.device | str = "cpu") -> "Population":
        """Discrete slots uniform over their levels; continuous slots zero."""
        return Population(
            self,
            {
                s.name: torch.zeros((n, s.dim), device=device)
                if s.levels is None
                else torch.randint(0, s.levels, (n, s.dim), device=device).float()
                for s in self.slots
            },
        )


def latent_schema(k: int, levels: int) -> StateSchema:
    """k unnamed variables z0..z{k-1}, in one slot."""
    variables = tuple(f"z{i}" for i in range(k))
    return StateSchema(slots=(Slot(name="z", variables=variables, levels=levels),))


@dataclass
class Population:
    """`values[slot]` has shape (n_cells, slot.dim)."""

    schema: StateSchema
    values: dict[str, torch.Tensor]

    @property
    def n(self) -> int:
        return int(next(iter(self.values.values())).shape[0])

    def get(self, variable: str) -> torch.Tensor:
        slot, i = self.schema.locate(variable)
        return self.values[slot][:, i]

    @property
    def device(self) -> torch.device:
        return next(iter(self.values.values())).device

    def copy(self) -> "Population":
        return Population(self.schema, {k: v.clone() for k, v in self.values.items()})
