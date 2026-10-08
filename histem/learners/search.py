"""Representation-agnostic edit search, plus random AST mutation for logic programs.

`Proposer` is the seam for different induction strategies. Random mutation is the dumb
baseline; an LLM proposer (read the program text + per-condition scores, return an
edited program) plugs in at the same place.
"""

from __future__ import annotations

import ast
import copy
import random
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np

from histem.models.logic import LogicDynamics
from histem.suite import Scores, Suite
from histem.world import WorldModel


class Proposer(Protocol):
    def propose(self, world: WorldModel, scores: Scores, rng: np.random.Generator) -> WorldModel: ...


@dataclass
class SearchLog:
    accepted: list[tuple[int, float, str]] = field(default_factory=list)  # (iter, objective, note)


def hill_climb(
    world: WorldModel, suite: Suite, proposer: Proposer, iters: int, seed: int = 0,
    verbose: bool = False,
) -> tuple[WorldModel, Scores, SearchLog]:
    rng = np.random.default_rng(seed)
    best = suite.evaluate(world)
    log = SearchLog([(0, best.objective, "init")])
    for it in range(1, iters + 1):
        try:
            cand = proposer.propose(world, best, rng)
            scores = suite.evaluate(cand)
        except (SyntaxError, NameError, KeyError, ValueError):
            continue
        if scores.objective < best.objective and not suite.regressions(best, scores):
            world, best = cand, scores
            log.accepted.append((it, best.objective, getattr(cand.dynamics, "last_edit", "")))
            if verbose:
                print(f"[{it}] objective={best.objective:.4f}  {log.accepted[-1][2]}")
    return world, best, log


# --- random mutation of logic programs -------------------------------------------------


class _Mutator:
    def __init__(self, vocab: list[str], rng: random.Random):
        self.vocab, self.rng = vocab, rng

    def literal(self) -> ast.expr:
        name = self.rng.choice(self.vocab)
        op = self.rng.choice([ast.GtE(), ast.GtE(), ast.Lt()])
        k = self.rng.choice([1, 2]) if not name.startswith("in_") else 0.5
        return ast.Compare(ast.Name(name, ast.Load()), [op], [ast.Constant(k)])

    def mutate(self, expr: str) -> str:
        tree = ast.parse(expr, mode="eval")
        nodes = [n for n in ast.walk(tree.body)]
        target = self.rng.choice(nodes)
        ops = [self._grow]
        if isinstance(target, ast.Name):
            ops.append(self._rename)
        if isinstance(target, ast.Constant):
            ops.append(self._bump)
        if isinstance(target, ast.BoolOp):
            ops += [self._flip_boolop, self._drop]
        if isinstance(target, (ast.Compare, ast.BoolOp, ast.UnaryOp)):
            ops.append(self._negate)
        new = self.rng.choice(ops)(tree, target)
        ast.fix_missing_locations(new)
        return ast.unparse(new.body)

    def _rename(self, tree, node):
        node.id = self.rng.choice(self.vocab)
        return tree

    def _bump(self, tree, node):
        if isinstance(node.value, bool):
            node.value = not node.value
        else:
            node.value = max(0, node.value + self.rng.choice([-1, 1]))
        return tree

    def _flip_boolop(self, tree, node):
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        return tree

    def _drop(self, tree, node):
        node.values.pop(self.rng.randrange(len(node.values)))
        if len(node.values) == 1:
            return _replace_node(tree, node, node.values[0])
        return tree

    def _negate(self, tree, node):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return _replace_node(tree, node, node.operand)
        return _replace_node(tree, node, ast.UnaryOp(ast.Not(), copy.deepcopy(node)))

    def _grow(self, tree, _node):
        """Combine the whole rule with a new literal, e.g. `e` -> `e and X >= 1`,
        or turn a boolean rule into a graded one: `2 if e else 0`."""
        body = tree.body
        choice = self.rng.random()
        if choice < 0.2 and not isinstance(body, ast.IfExp):
            tree.body = ast.IfExp(body, ast.Constant(2), ast.Constant(0))
        else:
            op = ast.And() if choice < 0.6 else ast.Or()
            tree.body = ast.BoolOp(op, [body, self.literal()])
        return tree


def _replace_node(tree, old, new):
    if tree.body is old:
        tree.body = new
        return tree
    for parent in ast.walk(tree):
        for fname, value in ast.iter_fields(parent):
            if value is old:
                setattr(parent, fname, new)
                return tree
            if isinstance(value, list) and any(v is old for v in value):
                setattr(parent, fname, [new if v is old else v for v in value])
                return tree
    return tree


@dataclass
class RandomLogicEdit:
    """Mutate one rule of a LogicDynamics at random."""

    seed: int = 0

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def propose(self, world: WorldModel, scores: Scores, rng: np.random.Generator) -> WorldModel:
        dyn: LogicDynamics = world.dynamics
        targets = list(dyn.rules) + [f"emit:{s}" for s in dyn.signal_rules]
        target = self._rng.choice(targets)
        mut = _Mutator(sorted(dyn.vocabulary), self._rng)
        if target.startswith("emit:"):
            sig = target[5:]
            signal_rules = {**dyn.signal_rules, sig: mut.mutate(dyn.signal_rules[sig])}
            new = LogicDynamics(dyn.schema, dyn.rules, signal_rules, dyn.rates, dyn.default_rate)
            new.last_edit = f"emit {sig} <- {signal_rules[sig]}"
        else:
            expr = mut.mutate(dyn.rules[target])
            new = dyn.with_rule(target, expr)
            new.last_edit = f"{target} <- {expr}"
        return replace(world, dynamics=new)
