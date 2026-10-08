from dataclasses import replace

import numpy as np
import pytest

from histem import Intervention, Slot, StateSchema, simulate, synthetic
from histem.dynamics import Dynamics
from histem.learners.search import RandomLogicEdit, hill_climb
from histem.models.logic import LogicDynamics, compile_expr
from histem.simulator import Signaling, knn_graph
from histem.suite import Suite


def test_logic_dynamics_satisfies_protocol():
    assert isinstance(synthetic.make_world().dynamics, Dynamics)


def test_expression_compiler_is_vectorised():
    f = compile_expr("2 if (a >= 1 and not b >= 2) or max(a, b) == 0 else 0", {"a", "b"})
    out = f({"a": np.array([1, 0, 1, 0]), "b": np.array([0, 0, 2, 1])})
    assert out.tolist() == [2, 2, 0, 0]


def test_unknown_names_rejected():
    with pytest.raises(NameError):
        compile_expr("a and c", {"a", "b"})
    with pytest.raises(SyntaxError):
        compile_expr("__import__('os')", {"a"})


def test_program_text_round_trips():
    dyn = synthetic.make_world().dynamics
    again = LogicDynamics.from_text(synthetic.SCHEMA, dyn.to_text())
    assert again.rules == dyn.rules and again.signal_rules == dyn.signal_rules
    assert again.rates == dyn.rates


def test_clamp_holds_through_simulation():
    w = synthetic.make_world()
    rng = np.random.default_rng(0)
    pop = w.sample_cells(200, rng, Intervention("ko", (("GATA1", 0),)))
    assert (pop.get("GATA1") == 0).all()


def test_discrete_values_stay_in_range():
    w = synthetic.make_world()
    pop = w.sample_cells(500, np.random.default_rng(0))
    for slot in synthetic.SCHEMA.slots:
        v = pop.values[slot.name]
        assert v.min() >= 0 and v.max() <= slot.levels - 1


def test_paracrine_signal_reaches_neighbors():
    schema = StateSchema((Slot("s", ("src", "rcv"), levels=2),))
    dyn = LogicDynamics(schema, {"rcv": "in_L >= 0.5"}, {"L": "1.0 if src >= 1 else 0.0"},
                        default_rate=1.0)
    pop = schema.empty(3)
    pop.values["s"][0, 0] = 1  # only cell 0 secretes
    positions = np.array([[0.0], [1.0], [10.0]])
    sig = Signaling(autocrine=0.0, paracrine=1.0, neighbors=knn_graph(positions, k=1))
    out, _ = simulate(dyn, pop, 1, np.random.default_rng(0), signaling=sig)
    assert out.get("rcv").tolist() == [0, 1, 0]  # cell 1's nearest neighbor is cell 0


def test_truth_beats_null_and_search_improves():
    truth = synthetic.make_world()
    suite = Suite([synthetic.make_dataset(truth, cells_per_condition=150)], cells_per_condition=150)
    null = LogicDynamics(synthetic.SCHEMA, {v: v for v in synthetic.SCHEMA.variables}, {"IL": "0.0"})
    start = replace(truth, dynamics=null)
    s_truth, s_null = suite.evaluate(truth), suite.evaluate(start)
    assert s_truth.objective < s_null.objective
    _, found, _ = hill_climb(start, suite, RandomLogicEdit(0), iters=30)
    assert found.objective <= s_null.objective
