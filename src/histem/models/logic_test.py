import pytest
import torch
from pydantic import ValidationError

from histem import synthetic
from histem.dynamics import Dynamics, Inputs
from histem.models.logic import LogicDynamics, RuleError, compile_expr
from histem.state import Slot, StateSchema

SCHEMA = StateSchema(slots=(Slot(name="s", variables=("a", "b"), levels=3),))


def test_logic_dynamics_satisfies_protocol() -> None:
    assert isinstance(synthetic.make_system().dynamics, Dynamics)


def test_expression_compiler_is_vectorised() -> None:
    rule = "2 if (a >= 1 and not b >= 2) or max(a, b) == 0 else 0"
    f = compile_expr(rule, {"a", "b"})
    out = f({"a": torch.tensor([1, 0, 1, 0]), "b": torch.tensor([0, 0, 2, 1])})
    assert out.tolist() == [2, 2, 0, 0]


@pytest.mark.parametrize("expr", ["a and c", "__import__('os')", "a >=", "a.b"])
def test_bad_expressions_raise_rule_error(expr: str) -> None:
    with pytest.raises(RuleError):
        compile_expr(expr, {"a", "b"})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rules": {"zzz": "a"}}, "not in schema"),
        ({"rules": {"a": "b"}, "rates": {"b": 0.5}}, "without rules"),
        ({"rules": {"a": "b"}, "rates": {"a": 1.5}}, r"\(0, 1\]"),
        ({"rules": {"a": "nope"}}, "unknown variable"),
    ],
)
def test_invalid_programs_rejected(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        LogicDynamics.model_validate({"state_schema": SCHEMA, **kwargs})


def test_program_text_round_trips() -> None:
    dyn = synthetic.make_system().dynamics
    assert isinstance(dyn, LogicDynamics)
    again = LogicDynamics.from_text(synthetic.SCHEMA, dyn.to_text())
    assert again.rules == dyn.rules
    assert again.signal_rules == dyn.signal_rules
    assert again.rates == dyn.rates


def test_unparseable_program_line_rejected() -> None:
    with pytest.raises(RuleError):
        LogicDynamics.from_text(SCHEMA, "a = b")


def test_with_rules_revalidates() -> None:
    dyn = LogicDynamics(state_schema=SCHEMA, rules={"a": "b"})
    assert dyn.with_rules(rules={"b": "a"}).rules == {"a": "b", "b": "a"}
    with pytest.raises(ValidationError):
        dyn.with_rules(rules={"a": "nope"})


def test_step_moves_one_level_toward_target() -> None:
    dyn = LogicDynamics(state_schema=SCHEMA, rules={"a": "2"}, default_rate=1.0)
    pop = SCHEMA.empty(4)
    out = dyn.step(pop, Inputs(torch.zeros((4, 0))))
    assert out.get("a").tolist() == [1, 1, 1, 1]
    assert pop.get("a").tolist() == [0, 0, 0, 0]  # input not mutated
