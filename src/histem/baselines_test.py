import pytest
import torch

from histem import baselines
from histem.data import CountsDataset
from histem.rng import seeded


def test_additive_beats_control_on_an_additive_double(ds: CountsDataset) -> None:
    assert baselines.additive(ds, "A_B", 20) < baselines.control(ds, "A_B", 20)


def test_additive_shift_of_a_single_is_its_own_shift(ds: CountsDataset) -> None:
    expected = ds.normalized("A").mean(0) - ds.normalized("control").mean(0)
    assert (baselines.additive_shift(ds, "A") - expected).abs().max() < 1e-6


def test_additive_needs_the_singles(ds: CountsDataset) -> None:
    _, only_double = ds.split(["control", "A_B"])
    with pytest.raises(KeyError, match="no single"):
        baselines.additive_shift(only_double, "A_B")


def test_columns_restrict_the_score(ds: CountsDataset) -> None:
    with seeded(0):
        full = baselines.control(ds, "A_B", 20)
    with seeded(0):
        all_columns = baselines.control(ds, "A_B", 20, list(range(len(ds.features))))
    with seeded(0):
        one = baselines.additive(ds, "A_B", 20, [0])
    assert torch.isclose(full, all_columns)
    assert one < full
