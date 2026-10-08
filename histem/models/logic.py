"""Multi-valued asynchronous logic programs: the first Dynamics implementation.

A program is plain text, one rule per line, so that people, LLMs, and mutation operators
can all read and edit it:

    GATA1 <- 2 if (GATA1 >= 1 or acc_GATA1 >= 1) and not PU1 >= 2 else 0
    acc_GATA1 <- in_EPO >= 0.5
    emit EPO <- 0.0
    rate acc_GATA1 = 0.05

`X <- expr` sets the *target* level of X. Each step, every ruled variable updates with
probability `rate`, moving one level toward its target (continuous variables relax toward
it). Variables without a rule hold their value. `in_<SIG>` is the received level of signal
SIG, and `emit SIG <- expr` is what the cell secretes.

Expressions are a vectorised Python subset: and/or/not, comparisons, + - *, x if c else y,
min/max, and numeric constants.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from histem.dynamics import Inputs, Intervention
from histem.state import Population, StateSchema

Env = dict[str, np.ndarray]
Compiled = Callable[[Env], np.ndarray]

_CMP = {
    ast.GtE: np.greater_equal, ast.Gt: np.greater, ast.LtE: np.less_equal,
    ast.Lt: np.less, ast.Eq: np.equal, ast.NotEq: np.not_equal,
}
_BIN = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply}


def compile_expr(source: str | ast.expr, names: set[str]) -> Compiled:
    node = ast.parse(source, mode="eval").body if isinstance(source, str) else source
    return _compile(node, names)


def _compile(node: ast.expr, names: set[str]) -> Compiled:
    match node:
        case ast.Constant(value=v) if isinstance(v, (int, float, bool)):
            return lambda env: np.float32(v)
        case ast.Name(id=name):
            if name not in names:
                raise NameError(f"unknown variable {name!r}")
            return lambda env: env[name]
        case ast.BoolOp(op=op, values=values):
            parts = [_compile(v, names) for v in values]
            f = np.logical_and if isinstance(op, ast.And) else np.logical_or
            return lambda env: f.reduce([np.asarray(p(env), bool) for p in parts])
        case ast.UnaryOp(op=ast.Not(), operand=x):
            p = _compile(x, names)
            return lambda env: np.logical_not(p(env))
        case ast.UnaryOp(op=ast.USub(), operand=x):
            p = _compile(x, names)
            return lambda env: -np.asarray(p(env), np.float32)
        case ast.Compare(left=left, ops=ops, comparators=comps):
            terms = [_compile(left, names)] + [_compile(c, names) for c in comps]
            fs = [_CMP[type(o)] for o in ops]

            def cmp(env):
                vals = [t(env) for t in terms]
                out = fs[0](vals[0], vals[1])
                for f, a, b in zip(fs[1:], vals[1:], vals[2:]):
                    out = out & f(a, b)
                return out

            return cmp
        case ast.BinOp(left=a, op=op, right=b) if type(op) in _BIN:
            f, pa, pb = _BIN[type(op)], _compile(a, names), _compile(b, names)
            return lambda env: f(np.asarray(pa(env), np.float32), np.asarray(pb(env), np.float32))
        case ast.IfExp(test=t, body=b, orelse=e):
            pt, pb, pe = (_compile(x, names) for x in (t, b, e))
            return lambda env: np.where(pt(env), pb(env), pe(env))
        case ast.Call(func=ast.Name(id=fn), args=args) if fn in ("min", "max") and args:
            ps = [_compile(a, names) for a in args]
            f = np.minimum if fn == "min" else np.maximum
            return lambda env: f.reduce([np.asarray(p(env), np.float32) for p in ps])
    raise SyntaxError(f"unsupported expression: {ast.unparse(node)}")


def expr_size(source: str) -> int:
    return sum(1 for n in ast.walk(ast.parse(source, mode="eval").body)
               if not isinstance(n, (ast.Load, ast.boolop, ast.cmpop, ast.operator, ast.unaryop)))


@dataclass
class LogicDynamics:
    schema: StateSchema
    rules: dict[str, str]  # target variable -> expression
    signal_rules: dict[str, str] = field(default_factory=dict)  # signal -> emitted amount
    rates: dict[str, float] = field(default_factory=dict)
    default_rate: float = 0.5

    def __post_init__(self):
        self.signal_names = tuple(self.signal_rules)
        names = self.vocabulary
        unknown = set(self.rules) - set(self.schema.variables)
        if unknown:
            raise KeyError(f"rules for variables not in schema: {sorted(unknown)}")
        self._rules = {v: compile_expr(e, names) for v, e in self.rules.items()}
        self._signals = {s: compile_expr(e, names) for s, e in self.signal_rules.items()}

    @property
    def vocabulary(self) -> set[str]:
        return set(self.schema.variables) | {f"in_{s}" for s in self.signal_names}

    # --- Dynamics protocol -------------------------------------------------------------

    def step(self, pop: Population, inputs: Inputs, rng: np.random.Generator) -> Population:
        env: Env = pop.variable_view()
        for j, s in enumerate(self.signal_names):
            env[f"in_{s}"] = inputs.signals[:, j]
        targets = {v: np.broadcast_to(f(env), (pop.n,)) for v, f in self._rules.items()}

        out = pop.copy()
        for v, target in targets.items():
            slot_name, i = self.schema.locate(v)
            slot = self.schema.slot(slot_name)
            fire = rng.random(pop.n) < self.rates.get(v, self.default_rate)
            cur = pop.values[slot_name][:, i]
            if slot.kind == "discrete":
                tgt = np.clip(np.rint(np.asarray(target, np.float32)), 0, slot.levels - 1)
                new = cur + np.sign(tgt - cur).astype(cur.dtype)
            else:
                new = cur + 0.5 * (np.asarray(target, np.float32) - cur)
            out.values[slot_name][:, i] = np.where(fire, new, cur)
        return out

    def emit_signals(self, pop: Population) -> np.ndarray:
        if not self.signal_names:
            return np.zeros((pop.n, 0), np.float32)
        env: Env = pop.variable_view()
        # emission sees no received signals; feed zeros so rules may still reference in_*
        for s in self.signal_names:
            env[f"in_{s}"] = np.zeros(pop.n, np.float32)
        return np.stack(
            [np.clip(np.broadcast_to(f(env), (pop.n,)), 0, None) for f in self._signals.values()],
            axis=1,
        ).astype(np.float32)

    def intervene(self, intervention: Intervention) -> LogicDynamics:
        return self

    def description_length(self) -> float:
        exprs = [*self.rules.values(), *self.signal_rules.values()]
        size = sum(expr_size(e) + 1 for e in exprs)  # +1 for naming the rule's target
        vocab = len(self.vocabulary) + 16  # variables + operators/constants
        return float(size * math.log2(vocab))

    # --- text form ---------------------------------------------------------------------

    def with_rule(self, target: str, expr: str) -> LogicDynamics:
        rules = dict(self.rules)
        rules[target] = expr
        return LogicDynamics(self.schema, rules, self.signal_rules, self.rates, self.default_rate)

    def to_text(self) -> str:
        lines = [f"{v} <- {e}" for v, e in self.rules.items()]
        lines += [f"emit {s} <- {e}" for s, e in self.signal_rules.items()]
        lines += [f"rate {v} = {r}" for v, r in self.rates.items()]
        return "\n".join(lines)

    @classmethod
    def from_text(cls, schema: StateSchema, text: str, default_rate: float = 0.5) -> LogicDynamics:
        rules, signals, rates = {}, {}, {}
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
            else:
                v, e = line.split("<-", 1)
                rules[v.strip()] = e.strip()
        return cls(schema, rules, signals, rates, default_rate)
