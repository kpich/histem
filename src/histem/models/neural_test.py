from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from histem import synthetic
from histem.models.neural import NeuralDynamics
from histem.rng import seeded
from histem.state import Slot, StateSchema
from histem.suite import Suite
from histem.system import CellSystem


def _system() -> CellSystem:
    torch.manual_seed(0)
    return synthetic.make_neural_system(hidden=8).model_copy(update={"burn_in": 3})


def test_rejects_discrete_slots() -> None:
    schema = StateSchema(slots=(Slot(name="s", variables=("a",)),))
    with pytest.raises(ValidationError, match="continuous"):
        NeuralDynamics(state_schema=schema)


def test_gradients_reach_every_parameter_through_clamps() -> None:
    system = _system()
    ds = synthetic.make_dataset(synthetic.make_system(), cells_per_condition=10)
    _, ko = ds.split(["GATA1_KO+PU1_KO"])
    Suite(datasets=[ko], cells_per_condition=10).loss(system).backward()
    for p in system.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0


def test_save_load_round_trip(tmp_path: Path) -> None:
    system = _system()
    system.save(tmp_path)
    loaded = CellSystem.load(tmp_path)
    with torch.no_grad(), seeded(0):
        before = system.sample(20)
    with torch.no_grad(), seeded(0):
        after = loaded.sample(20)
    assert torch.equal(before, after)


def test_to_copies_parameters() -> None:
    system = _system()
    moved = system.to("cpu")
    for a, b in zip(system.parameters(), moved.parameters(), strict=True):
        assert a is not b
        assert torch.equal(a, b)
        assert b.requires_grad
