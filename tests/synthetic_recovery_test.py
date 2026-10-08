"""Integration: generate synthetic Perturb-seq, score programs, run search."""

from dataclasses import replace

from histem import synthetic
from histem.learners.search import RandomLogicEdit, hill_climb
from histem.models.logic import LogicDynamics
from histem.suite import Suite


def test_truth_beats_null_and_search_improves() -> None:
    truth = synthetic.make_world()
    data = synthetic.make_dataset(truth, cells_per_condition=150)
    suite = Suite([data], cells_per_condition=150)
    null = LogicDynamics(
        synthetic.SCHEMA, {v: v for v in synthetic.SCHEMA.variables}, {"IL": "0.0"}
    )
    start = replace(truth, dynamics=null)
    s_truth, s_null = suite.evaluate(truth), suite.evaluate(start)
    assert s_truth.objective < s_null.objective
    _, found, _ = hill_climb(start, suite, RandomLogicEdit(0), iters=30)
    assert found.objective <= s_null.objective
