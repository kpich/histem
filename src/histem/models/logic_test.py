import numpy as np
import pytest

from histem import synthetic
from histem.dynamics import Dynamics
from histem.models.logic import LogicDynamics, compile_expr


def test_logic_dynamics_satisfies_protocol() -> None:
    assert isinstance(synthetic.make_world().dynamics, Dynamics)


def test_expression_compiler_is_vectorised() -> None:
    f = compile_expr(
        "2 if (a >= 1 and not b >= 2) or max(a, b) == 0 else 0", {"a", "b"}
    )
    out = f({"a": np.array([1, 0, 1, 0]), "b": np.array([0, 0, 2, 1])})
    assert out.tolist() == [2, 2, 0, 0]


def test_unknown_names_rejected() -> None:
    with pytest.raises(NameError):
        compile_expr("a and c", {"a", "b"})


def test_unsupported_syntax_rejected() -> None:
    with pytest.raises(SyntaxError):
        compile_expr("__import__('os')", {"a"})


def test_program_text_round_trips() -> None:
    dyn = synthetic.make_world().dynamics
    assert isinstance(dyn, LogicDynamics)
    again = LogicDynamics.from_text(synthetic.SCHEMA, dyn.to_text())
    assert again.rules == dyn.rules
    assert again.signal_rules == dyn.signal_rules
    assert again.rates == dyn.rates
