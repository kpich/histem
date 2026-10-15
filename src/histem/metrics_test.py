import torch

from histem.metrics import energy_distance, probe_accuracy
from histem.rng import seeded


def test_energy_distance_zero_on_identical_and_grows_with_shift() -> None:
    x = torch.randn((200, 3), generator=torch.Generator().manual_seed(0))
    assert energy_distance(x, x) == 0.0
    assert energy_distance(x, x + 0.5) < energy_distance(x, x + 2.0)


def test_probe_separates_shifted_groups_and_not_identical_ones() -> None:
    with seeded(0):
        x = torch.randn(400, 5)
        apart = probe_accuracy({"a": x[:200], "b": x[200:] + 3.0})
        same = probe_accuracy({"a": x[:200], "b": x[200:]})
    assert apart > 0.95
    assert same < 0.7
