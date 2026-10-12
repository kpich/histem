import pytest
import torch

from histem import baselines
from histem.data import CountsDataset
from histem.dynamics import CONTROL, Intervention


def _ds() -> CountsDataset:
    torch.manual_seed(0)
    base = torch.full((50, 3), 10.0)
    return CountsDataset(
        name="d",
        features=("a", "b", "c"),
        counts={
            "control": torch.poisson(base),
            "A": torch.poisson(base * torch.tensor([4.0, 1, 1])),
            "B": torch.poisson(base * torch.tensor([1.0, 4, 1])),
            "A_B": torch.poisson(base * torch.tensor([4.0, 4, 1])),
        },
        interventions={
            "control": CONTROL,
            "A": Intervention(name="A", genes=("A",)),
            "B": Intervention(name="B", genes=("B",)),
            "A_B": Intervention(name="A_B", genes=("A", "B")),
        },
    )


def test_additive_beats_control_on_an_additive_double() -> None:
    ds = _ds()
    assert baselines.additive(ds, "A_B", 50) < baselines.control(ds, "A_B", 50)


def test_additive_shift_of_a_single_is_its_own_shift() -> None:
    ds = _ds()
    expected = ds.normalized("A").mean(0) - ds.normalized("control").mean(0)
    assert torch.allclose(baselines.additive_shift(ds, "A"), expected)


def test_additive_needs_the_singles() -> None:
    ds = _ds()
    _, only_double = ds.split(["control", "A_B"])
    with pytest.raises(KeyError, match="no single"):
        baselines.additive_shift(only_double, "A_B")
