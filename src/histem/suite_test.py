from pathlib import Path

import pytest

from histem import synthetic
from histem.data import CountsDataset
from histem.suite import Suite


def test_objective_weights_datasets_not_conditions() -> None:
    system = synthetic.make_system().model_copy(update={"burn_in": 2})
    ds = synthetic.make_dataset(system, cells_per_condition=10)
    big, small = ds.split(["control"])
    suite = Suite(datasets=[big, small], cells_per_condition=10)
    per_ds = suite.evaluate(system).by_dataset()
    heavy = suite.model_copy(update={"weights": {small.name: 3.0}}).evaluate(system)
    expected = (per_ds[big.name] + 3 * per_ds[small.name]) / 4
    assert heavy.objective == pytest.approx(expected)


def test_save_load_round_trip(tmp_path: Path) -> None:
    system = synthetic.make_system().model_copy(update={"burn_in": 2})
    ds = synthetic.make_dataset(system, cells_per_condition=10)
    suite = Suite(datasets=[ds], cells_per_condition=10, weights={ds.name: 2.0})
    suite.save(tmp_path)
    loaded = Suite.load(tmp_path)
    assert isinstance(loaded.datasets[0], CountsDataset)
    assert loaded.evaluate(system) == suite.evaluate(system)
