import torch

from histem.metrics import energy_distance


def test_energy_distance_zero_on_identical_and_grows_with_shift() -> None:
    x = torch.randn((200, 3), generator=torch.Generator().manual_seed(0))
    assert energy_distance(x, x) == 0.0
    assert energy_distance(x, x + 0.5) < energy_distance(x, x + 2.0)
