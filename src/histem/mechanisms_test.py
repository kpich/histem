import pytest
import torch
from pydantic import ValidationError

from histem.dynamics import Inputs
from histem.mechanisms import Composite
from histem.spec import FrozenSpec
from histem.state import Population, Slot

A = Slot(name="a", variables=("a0",), levels=None)
B = Slot(name="b", variables=("b0",), levels=None)


class Add(FrozenSpec):
    """`dst` += `src` + 1."""

    src: Slot
    dst: Slot

    @property
    def writes(self) -> tuple[Slot, ...]:
        return (self.dst,)

    @property
    def reads(self) -> tuple[str, ...]:
        return (self.src.name, self.dst.name)

    def step(self, pop: Population, inputs: Inputs) -> dict[str, torch.Tensor]:
        return {
            self.dst.name: pop.values[self.dst.name] + pop.values[self.src.name] + 1
        }


def test_mechanisms_run_in_order_and_see_earlier_writes() -> None:
    dyn = Composite(mechanisms={"a": Add(src=B, dst=A), "b": Add(src=A, dst=B)})
    assert [s.name for s in dyn.state_schema.slots] == ["a", "b"]
    pop = dyn.state_schema.empty(2)
    out = dyn.step(pop, Inputs(torch.zeros(2, 0)))
    assert out.values["a"].tolist() == [[1.0], [1.0]]
    assert out.values["b"].tolist() == [[2.0], [2.0]]
    assert pop.values["a"].sum() == 0


def test_unwritten_reads_rejected() -> None:
    with pytest.raises(ValidationError, match="no mechanism writes"):
        Composite(mechanisms={"a": Add(src=B, dst=A)})


def test_conflicting_slots_rejected() -> None:
    discrete = Slot(name="a", variables=("a0",), levels=3)
    with pytest.raises(ValidationError, match="conflicts"):
        Composite(
            mechanisms={"a": Add(src=A, dst=A), "b": Add(src=discrete, dst=discrete)}
        )
