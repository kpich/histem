import math

from histem import synthetic
from histem.learners.search import RandomLogicEdit, hill_climb
from histem.models.logic import LogicDynamics
from histem.suite import Suite


def test_generate_score_search_round_trip() -> None:
    system = synthetic.make_system().model_copy(update={"burn_in": 3})
    train, test = synthetic.make_dataset(system, cells_per_condition=10).split(
        ["GATA1_KO"]
    )
    suite = Suite(datasets=[train], cells_per_condition=10)

    scores = suite.evaluate(system)
    assert set(scores.fit) == {(train.name, c) for c in train.conditions}
    assert all(math.isfinite(v) for v in scores.fit.values())
    assert test.conditions == ["GATA1_KO"]

    found, best, log = hill_climb(system, suite, RandomLogicEdit(seed=0), iters=3)
    assert best.objective <= scores.objective
    assert log.accepted[0][2] == "init"
    assert isinstance(found.dynamics, LogicDynamics)
    reparsed = LogicDynamics.from_text(synthetic.SCHEMA, found.dynamics.to_text())
    assert reparsed.rules == found.dynamics.rules

    counts = found.sample(5)
    assert counts.shape == (5, len(train.features))
