from pathlib import Path

import torch

from histem import synthetic
from histem.dynamics import Intervention
from histem.models.logic import LogicDynamics
from histem.rng import seeded
from histem.system import CellSystem, UniformInit


def test_clamp_holds_through_simulation() -> None:
    w = synthetic.make_system()
    ko = Intervention(name="ko", clamps=(("GATA1", 0),))
    pop = w.sample_cells(200, ko)
    assert (pop.get("GATA1") == 0).all()


def test_discrete_values_stay_in_range() -> None:
    torch.manual_seed(0)
    pop = synthetic.make_system().sample_cells(500)
    for slot in synthetic.SCHEMA.slots:
        v = pop.values[slot.name]
        assert v.min() >= 0
        assert v.max() <= slot.levels - 1


def test_save_load_round_trip(tmp_path: Path) -> None:
    system = synthetic.make_system().model_copy(update={"burn_in": 5})
    system.save(tmp_path)
    loaded = CellSystem.load(tmp_path)
    assert isinstance(loaded.dynamics, LogicDynamics)
    assert isinstance(loaded.init, UniformInit)
    with seeded(0):
        before = system.sample(50)
    with seeded(0):
        after = loaded.sample(50)
    assert torch.equal(before, after)
