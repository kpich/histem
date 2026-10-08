import pytest

from histem import synthetic
from histem.suite import Scores, Suite


def _scores(**fit: float) -> Scores:
    f = {(k, "c"): v for k, v in fit.items()}
    return Scores(fit=f, description_length=0.0, objective=sum(fit.values()))


def test_regressions_flags_only_datasets_beyond_tolerance() -> None:
    before = _scores(a=1.0, b=1.0)
    after = _scores(a=1.04, b=1.2)
    assert Suite.regressions(before, after, tolerance=0.05) == ["b"]


def test_objective_weights_datasets_not_conditions() -> None:
    system = synthetic.make_system().model_copy(update={"burn_in": 2})
    ds = synthetic.make_dataset(system, cells_per_condition=10)
    big, small = ds.split(["control"])
    suite = Suite(datasets=[big, small], cells_per_condition=10, complexity_weight=0)
    per_ds = suite.evaluate(system).by_dataset()
    heavy = suite.model_copy(update={"weights": {small.name: 3.0}}).evaluate(system)
    expected = (per_ds[big.name] + 3 * per_ds[small.name]) / 4
    assert heavy.objective == pytest.approx(expected)
