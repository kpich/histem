"""Multi-valued asynchronous logic programs, written as plain text:

    GATA1 <- 2 if (GATA1 >= 1 or acc_GATA1 >= 1) and not PU1 >= 2 else 0
    acc_GATA1 <- in_EPO >= 0.5
    emit EPO <- 0.0
    rate acc_GATA1 = 0.05

`X <- expr` sets X's target. Each step, each ruled variable fires with probability
`rate` and moves one level toward its target. Unruled variables hold. `in_S` is the
received level of signal S. Expressions are a vectorised Python subset.
"""

import ast
import functools
import math
from collections.abc import Callable
from typing import Self

import torch
from pydantic import Field, PrivateAttr, model_validator

from histem.dynamics import Inputs, Intervention
from histem.spec import FrozenSpec
from histem.state import Population, StateSchema

Env = dict[str, torch.Tensor]
Compiled = Callable[[Env], torch.Tensor]

_CMP = {
    ast.GtE: torch.ge,
    ast.Gt: torch.gt,
    ast.LtE: torch.le,
    ast.Lt: torch.lt,
    ast.Eq: torch.eq,
    ast.NotEq: torch.ne,
}
_BIN: dict[type[ast.operator], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    ast.Add: torch.add,
    ast.Sub: torch.sub,
    ast.Mult: torch.mul,
}


def _device(env: Env) -> torch.device:
    return next(iter(env.values())).device


class RuleError(ValueError):
    pass


def compile_expr(source: str | ast.expr, names: set[str]) -> Compiled:
    if isinstance(source, str):
        try:
            source = ast.parse(source, mode="eval").body
        except SyntaxError as e:
            raise RuleError(f"cannot parse rule {source!r}: {e.msg}") from e
    return _compile(source, names)


def _compile(node: ast.expr, names: set[str]) -> Compiled:
    match node:
        case ast.Constant(value=v) if isinstance(v, (int, float, bool)):
            return lambda env: torch.tensor(float(v), device=_device(env))
        case ast.Name(id=name):
            if name not in names:
                raise RuleError(f"unknown variable {name!r}")
            return lambda env: env[name]
        case ast.BoolOp(op=op, values=values):
            parts = [_compile(v, names) for v in values]
            bool_f = torch.logical_and if isinstance(op, ast.And) else torch.logical_or
            return lambda env: functools.reduce(bool_f, [p(env).bool() for p in parts])
        case ast.UnaryOp(op=ast.Not(), operand=x):
            p = _compile(x, names)
            return lambda env: torch.logical_not(p(env))
        case ast.UnaryOp(op=ast.USub(), operand=x):
            p = _compile(x, names)
            return lambda env: -p(env).float()
        case ast.Compare(left=left, ops=ops, comparators=comps):
            terms = [_compile(left, names)] + [_compile(c, names) for c in comps]
            fs = [_CMP[type(o)] for o in ops]

            def cmp(env: Env) -> torch.Tensor:
                vals = [t(env) for t in terms]
                out = fs[0](vals[0], vals[1])
                for f, a, b in zip(fs[1:], vals[1:], vals[2:], strict=False):
                    out = out & f(a, b)
                return out

            return cmp
        case ast.BinOp(left=a, op=op, right=b) if type(op) in _BIN:
            bin_f, pa, pb = _BIN[type(op)], _compile(a, names), _compile(b, names)
            return lambda env: bin_f(pa(env).float(), pb(env).float())
        case ast.IfExp(test=t, body=b, orelse=e):
            pt, pb, pe = (_compile(x, names) for x in (t, b, e))
            return lambda env: torch.where(
                pt(env).bool(), pb(env).float(), pe(env).float()
            )
        case ast.Call(func=ast.Name(id=fn), args=args) if fn in ("min", "max") and args:
            ps = [_compile(a, names) for a in args]
            mm_f = torch.minimum if fn == "min" else torch.maximum
            return lambda env: functools.reduce(mm_f, [p(env).float() for p in ps])
    raise RuleError(f"unsupported expression: {ast.unparse(node)}")


def expr_size(source: str) -> int:
    return sum(
        1
        for n in ast.walk(ast.parse(source, mode="eval").body)
        if not isinstance(
            n, (ast.Load, ast.boolop, ast.cmpop, ast.operator, ast.unaryop)
        )
    )


class LogicDynamics(FrozenSpec):
    state_schema: StateSchema
    rules: dict[str, str]
    signal_rules: dict[str, str] = Field(default_factory=dict)
    rates: dict[str, float] = Field(default_factory=dict)
    default_rate: float = Field(0.5, gt=0, le=1)

    _rules: dict[str, Compiled] = PrivateAttr()
    _signals: dict[str, Compiled] = PrivateAttr()

    @model_validator(mode="after")
    def _compile_rules(self) -> Self:
        variables = set(self.state_schema.variables)
        if unknown := set(self.rules) - variables:
            raise ValueError(f"rules for variables not in schema: {sorted(unknown)}")
        if unknown := set(self.rates) - set(self.rules):
            raise ValueError(f"rates for variables without rules: {sorted(unknown)}")
        if bad := {v: r for v, r in self.rates.items() if not 0 < r <= 1}:
            raise ValueError(f"rates must be in (0, 1]: {bad}")
        names = self.vocabulary
        self._rules = {v: compile_expr(e, names) for v, e in self.rules.items()}
        self._signals = {
            s: compile_expr(e, names) for s, e in self.signal_rules.items()
        }
        return self

    @property
    def signal_names(self) -> tuple[str, ...]:
        return tuple(self.signal_rules)

    @property
    def vocabulary(self) -> set[str]:
        return set(self.state_schema.variables) | {f"in_{s}" for s in self.signal_names}

    def step(self, pop: Population, inputs: Inputs) -> Population:
        env: Env = pop.variable_view()
        for j, s in enumerate(self.signal_names):
            env[f"in_{s}"] = inputs.signals[:, j]
        targets = {v: f(env).float().expand(pop.n) for v, f in self._rules.items()}

        out = pop.copy()
        for v, target in targets.items():
            slot_name, i = self.state_schema.locate(v)
            slot = self.state_schema.slot(slot_name)
            cur = pop.values[slot_name][:, i]
            fire = torch.rand(pop.n, device=cur.device) < self.rates.get(
                v, self.default_rate
            )
            if slot.kind == "discrete":
                tgt = target.round().clamp(0, slot.levels - 1)
                new = cur + torch.sign(tgt - cur).to(cur.dtype)
            else:
                new = cur + 0.5 * (target - cur)
            out.values[slot_name][:, i] = torch.where(fire, new, cur)
        return out

    def emit_signals(self, pop: Population) -> torch.Tensor:
        if not self.signal_names:
            return torch.zeros((pop.n, 0), device=pop.device)
        env: Env = pop.variable_view()
        for s in self.signal_names:
            env[f"in_{s}"] = torch.zeros(pop.n, device=pop.device)
        return torch.stack(
            [f(env).float().expand(pop.n).clamp(min=0) for f in self._signals.values()],
            dim=1,
        )

    def intervene(self, intervention: Intervention) -> Self:
        return self

    def description_length(self) -> float:
        exprs = [*self.rules.values(), *self.signal_rules.values()]
        size = sum(expr_size(e) + 1 for e in exprs)
        vocab = len(self.vocabulary) + 16  # + operators and constants
        return float(size * math.log2(vocab))

    def with_rules(
        self,
        rules: dict[str, str] | None = None,
        signal_rules: dict[str, str] | None = None,
    ) -> "LogicDynamics":
        return LogicDynamics(
            state_schema=self.state_schema,
            rules={**self.rules, **(rules or {})},
            signal_rules={**self.signal_rules, **(signal_rules or {})},
            rates=self.rates,
            default_rate=self.default_rate,
        )

    def to_text(self) -> str:
        lines = [f"{v} <- {e}" for v, e in self.rules.items()]
        lines += [f"emit {s} <- {e}" for s, e in self.signal_rules.items()]
        lines += [f"rate {v} = {r}" for v, r in self.rates.items()]
        return "\n".join(lines)

    @classmethod
    def from_text(
        cls, schema: StateSchema, text: str, default_rate: float = 0.5
    ) -> Self:
        rules: dict[str, str] = {}
        signals: dict[str, str] = {}
        rates: dict[str, float] = {}
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("rate "):
                v, r = line[5:].split("=")
                rates[v.strip()] = float(r)
            elif line.startswith("emit "):
                s, e = line[5:].split("<-", 1)
                signals[s.strip()] = e.strip()
            elif "<-" in line:
                v, e = line.split("<-", 1)
                rules[v.strip()] = e.strip()
            else:
                raise RuleError(f"cannot parse program line: {raw!r}")
        return cls(
            state_schema=schema,
            rules=rules,
            signal_rules=signals,
            rates=rates,
            default_rate=default_rate,
        )
