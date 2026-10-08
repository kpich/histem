import numpy as np

from histem import synthetic
from histem.dynamics import Intervention


def test_clamp_holds_through_simulation() -> None:
    w = synthetic.make_system()
    ko = Intervention(name="ko", clamps=(("GATA1", 0),))
    pop = w.sample_cells(200, np.random.default_rng(0), ko)
    assert (pop.get("GATA1") == 0).all()


def test_discrete_values_stay_in_range() -> None:
    pop = synthetic.make_system().sample_cells(500, np.random.default_rng(0))
    for slot in synthetic.SCHEMA.slots:
        v = pop.values[slot.name]
        assert v.min() >= 0
        assert v.max() <= slot.levels - 1
