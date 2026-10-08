"""Representation-agnostic edit search, plus random AST mutation for logic programs.

`Proposer` is the seam for different induction strategies. Random mutation is the dumb
baseline; an LLM proposer (read the program text + per-condition scores, return an
edited program) plugs in at the same place.
"""

import ast
import copy
import random
from collections.abc import Callable
from typing import Protocol

import numpy as np
from pydantic import Field, PrivateAttr

from histem.models.logic import LogicDynamics
from histem.spec import FrozenSpec, Spec
from histem.suite import Scores, Suite
from histem.world import WorldModel


class Proposal(FrozenSpec):
    world: WorldModel
    note: str = ""  # human-readable description of the edit, for logs


class Proposer(Protocol):
    def propose(
        self, world: WorldModel, scores: Scores, rng: np.random.Generator
    ) -> Proposal: ...


class SearchLog(Spec):
    # (iteration, objective, note) for every accepted proposal
    accepted: list[tuple[int, float, str]] = Field(default_factory=list)


def hill_climb(
    world: WorldModel,
    suite: Suite,
    proposer: Proposer,
    iters: int,
    *,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[WorldModel, Scores, SearchLog]:
    rng = np.random.default_rng(seed)
    best = suite.evaluate(world)
    log = SearchLog(accepted=[(0, best.objective, "init")])
    for it in range(1, iters + 1):
        try:
            proposal = proposer.propose(world, best, rng)
            scores = suite.evaluate(proposal.world)
        except (KeyError, ValueError):  # invalid proposals (incl. RuleError)
            continue
        if scores.objective < best.objective and not Suite.regressions(best, scores):
            world, best = proposal.world, scores
            log.accepted.append((it, best.objective, proposal.note))
            if verbose:
                print(f"[{it}] objective={best.objective:.4f}  {proposal.note}")
    return world, best, log


# --- random mutation of logic programs ---------------------------------------


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
        target = self.rng.choice(list(ast.walk(tree.body)))
        ops: list[Callable[[], ast.Expression]] = [lambda: self._grow(tree)]
        match target:
            case ast.Name():
                ops.append(lambda: self._rename(tree, target))
            case ast.Constant():
                ops.append(lambda: self._bump(tree, target))
            case ast.BoolOp():
                ops.append(lambda: self._flip_boolop(tree, target))
                ops.append(lambda: self._drop(tree, target))
        if isinstance(target, (ast.Compare, ast.BoolOp, ast.UnaryOp)):
            ops.append(lambda: self._negate(tree, target))
        new = self.rng.choice(ops)()
        ast.fix_missing_locations(new)
        return ast.unparse(new.body)

    def _rename(self, tree: ast.Expression, node: ast.Name) -> ast.Expression:
        node.id = self.rng.choice(self.vocab)
        return tree

    def _bump(self, tree: ast.Expression, node: ast.Constant) -> ast.Expression:
        match node.value:
            case bool(v):
                node.value = not v
            case int(v) | float(v):
                node.value = max(0, v + self.rng.choice([-1, 1]))
        return tree

    def _flip_boolop(self, tree: ast.Expression, node: ast.BoolOp) -> ast.Expression:
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        return tree

    def _drop(self, tree: ast.Expression, node: ast.BoolOp) -> ast.Expression:
        node.values.pop(self.rng.randrange(len(node.values)))
        if len(node.values) == 1:
            return _replace_node(tree, node, node.values[0])
        return tree

    def _negate(self, tree: ast.Expression, node: ast.expr) -> ast.Expression:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return _replace_node(tree, node, node.operand)
        return _replace_node(tree, node, ast.UnaryOp(ast.Not(), copy.deepcopy(node)))

    def _grow(self, tree: ast.Expression) -> ast.Expression:
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


def _replace_node(tree: ast.Expression, old: ast.expr, new: ast.expr) -> ast.Expression:
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


class RandomLogicEdit(Spec):
    """Mutate one rule of a LogicDynamics at random."""

    seed: int = 0
    _rng: random.Random = PrivateAttr()

    def model_post_init(self, context: object) -> None:
        self._rng = random.Random(self.seed)

    def propose(
        self, world: WorldModel, scores: Scores, rng: np.random.Generator
    ) -> Proposal:
        dyn = world.dynamics
        if not isinstance(dyn, LogicDynamics):
            raise TypeError(f"RandomLogicEdit needs LogicDynamics, got {type(dyn)}")
        targets = list(dyn.rules) + [f"emit:{s}" for s in dyn.signal_rules]
        target = self._rng.choice(targets)
        mut = _Mutator(sorted(dyn.vocabulary), self._rng)
        if target.startswith("emit:"):
            sig = target.removeprefix("emit:")
            expr = mut.mutate(dyn.signal_rules[sig])
            new = dyn.with_rules(signal_rules={sig: expr})
            note = f"emit {sig} <- {expr}"
        else:
            expr = mut.mutate(dyn.rules[target])
            new = dyn.with_rules(rules={target: expr})
            note = f"{target} <- {expr}"
        return Proposal(world=world.with_dynamics(new), note=note)
