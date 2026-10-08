import math

import numpy as np

from histem import synthetic
from histem.learners.search import RandomLogicEdit, hill_climb
from histem.models.logic import LogicDynamics
from histem.suite import Suite


def test_generate_score_search_round_trip() -> None:
    world = synthetic.make_world().model_copy(update={"burn_in": 3})
    train, test = synthetic.make_dataset(world, cells_per_condition=10).split(
        ["GATA1_KO"]
    )
    suite = Suite(datasets=[train], cells_per_condition=10)

    scores = suite.evaluate(world)
    assert set(scores.fit) == {(train.name, c) for c in train.conditions}
    assert all(math.isfinite(v) for v in scores.fit.values())
    assert test.conditions == ["GATA1_KO"]

    found, best, log = hill_climb(world, suite, RandomLogicEdit(seed=0), iters=3)
    assert best.objective <= scores.objective
    assert log.accepted[0][2] == "init"
    assert isinstance(found.dynamics, LogicDynamics)
    reparsed = LogicDynamics.from_text(synthetic.SCHEMA, found.dynamics.to_text())
    assert reparsed.rules == found.dynamics.rules

    adata = found.sample(5, np.random.default_rng(0))
    assert adata.n_obs == 5
    assert adata.var_names.equals(train.adata.var_names)
