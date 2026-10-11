from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from histem import synthetic
from histem.dynamics import Intervention
from histem.learners.emulate import record
from histem.models.diffusion import DiffusionKernel
from histem.rng import seeded
from histem.suite import Suite
from histem.system import CellSystem


def _system() -> CellSystem:
    torch.manual_seed(0)
    return synthetic.make_emulator_system(width=16, layers=1).model_copy(
        update={"burn_in": 3, "grad_steps": 2}
    )


def test_rejects_discrete_slots() -> None:
    with pytest.raises(ValidationError, match="continuous"):
        DiffusionKernel(state_schema=synthetic.SCHEMA)


def test_steps_stay_on_each_variables_levels() -> None:
    pop = _system().sample_cells(200)
    for slot in pop.schema.slots:
        v = pop.values[slot.name]
        assert torch.equal(v, v.round())
        assert v.min() >= 0
        assert v.max() <= slot.levels - 1


def test_gradients_reach_every_parameter_through_clamps() -> None:
    system = _system()
    ds = synthetic.make_dataset(synthetic.make_system(), cells_per_condition=10)
    _, ko = ds.split(["GATA1_KO+PU1_KO"])
    Suite(datasets=[ko], cells_per_condition=10).loss(system).backward()
    for p in system.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_transition_loss_is_differentiable() -> None:
    system = _system()
    data = record(synthetic.make_system(), [Intervention()], 8, 2)
    kernel = system.dynamics
    assert isinstance(kernel, DiffusionKernel)
    loss = kernel.transition_loss(data)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in kernel.parameters())


def test_save_load_round_trip(tmp_path: Path) -> None:
    system = _system()
    system.save(tmp_path)
    loaded = CellSystem.load(tmp_path)
    with torch.no_grad(), seeded(0):
        before = system.sample(20)
    with torch.no_grad(), seeded(0):
        after = loaded.sample(20)
    assert torch.equal(before, after)
