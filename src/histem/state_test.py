import pytest

from histem.state import Slot, StateSchema


def test_duplicate_variable_names_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        StateSchema((Slot("a", ("x", "y")), Slot("b", ("y",))))


def test_locate_and_get() -> None:
    schema = StateSchema((Slot("a", ("x",)), Slot("b", ("y", "z"), levels=3)))
    assert schema.locate("z") == ("b", 1)
    pop = schema.empty(4)
    pop.values["b"][:, 1] = 2
    assert pop.get("z").tolist() == [2, 2, 2, 2]
