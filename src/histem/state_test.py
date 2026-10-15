import pytest
from pydantic import ValidationError

from histem.state import Slot, StateSchema


def test_duplicate_variable_names_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        StateSchema(
            slots=(
                Slot(name="a", variables=("x", "y")),
                Slot(name="b", variables=("y",)),
            )
        )


def test_levels_at_least_two() -> None:
    with pytest.raises(ValidationError, match="levels"):
        Slot(name="a", variables=("x",), levels=1)


def test_empty_slot_rejected() -> None:
    with pytest.raises(ValidationError):
        Slot(name="a", variables=())


def test_locate_and_get() -> None:
    schema = StateSchema(
        slots=(
            Slot(name="a", variables=("x",)),
            Slot(name="b", variables=("y", "z"), levels=3),
        )
    )
    assert schema.locate("z") == ("b", 1)
    pop = schema.empty(4)
    pop.values["b"][:, 1] = 2
    assert pop.get("z").tolist() == [2, 2, 2, 2]


def test_continuous_slots_start_at_zero() -> None:
    schema = StateSchema(
        slots=(
            Slot(name="a", variables=("x",), levels=None),
            Slot(name="b", variables=("y",)),
        )
    )
    pop = schema.uniform(100)
    assert pop.values["a"].abs().sum() == 0
    assert set(pop.values["b"].flatten().tolist()) == {0.0, 1.0}
