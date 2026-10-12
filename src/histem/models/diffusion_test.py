import pytest

from histem.dynamics import Intervention
from histem.system import CellSystem


def test_steps_stay_on_each_variables_levels(system: CellSystem) -> None:
    pop = system.sample_cells(200)
    for slot in pop.schema.slots:
        v = pop.values[slot.name]
        assert (v == v.round()).all()
        assert v.min() >= 0
        assert v.max() <= slot.levels - 1


def test_unknown_gene_rejected(system: CellSystem) -> None:
    with pytest.raises(KeyError, match="unknown genes"):
        system.sample_cells(2, Intervention(name="Z", genes=("Z",)))
