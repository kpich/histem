import numpy as np

from histem.metrics import energy_distance


def test_energy_distance_zero_on_identical_and_grows_with_shift() -> None:
    x = np.random.default_rng(0).standard_normal((200, 3))
    assert energy_distance(x, x) == 0.0
    assert energy_distance(x, x + 0.5) < energy_distance(x, x + 2.0)
