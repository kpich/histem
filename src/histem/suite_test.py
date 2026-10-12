from pathlib import Path

import pytest

from histem.data import CountsDataset
from histem.suite import Suite
from histem.system import CellSystem


def test_objective_weights_datasets_not_conditions(
    ds: CountsDataset, system: CellSystem
) -> None:
    big, small = ds.split(["control"])
    suite = Suite(datasets=[big, small], cells_per_condition=10)
    per_ds = suite.evaluate(system).by_dataset()
    heavy = suite.model_copy(update={"weights": {small.name: 3.0}}).evaluate(system)
    expected = (per_ds[big.name] + 3 * per_ds[small.name]) / 4
    assert heavy.objective == pytest.approx(expected)


def test_save_load_round_trip(
    tmp_path: Path, ds: CountsDataset, system: CellSystem
) -> None:
    suite = Suite(datasets=[ds], cells_per_condition=10, weights={ds.name: 2.0})
    suite.save(tmp_path)
    loaded = Suite.load(tmp_path)
    assert isinstance(loaded.datasets[0], CountsDataset)
    assert loaded.evaluate(system) == suite.evaluate(system)
