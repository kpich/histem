import pytest

from histem import baselines
from histem.data import CountsDataset


def test_additive_beats_control_on_an_additive_double(ds: CountsDataset) -> None:
    assert baselines.additive(ds, "A_B", 20) < baselines.control(ds, "A_B", 20)


def test_additive_shift_of_a_single_is_its_own_shift(ds: CountsDataset) -> None:
    expected = ds.normalized("A").mean(0) - ds.normalized("control").mean(0)
    assert (baselines.additive_shift(ds, "A") - expected).abs().max() < 1e-6


def test_additive_needs_the_singles(ds: CountsDataset) -> None:
    _, only_double = ds.split(["control", "A_B"])
    with pytest.raises(KeyError, match="no single"):
        baselines.additive_shift(only_double, "A_B")
