import pytest

from histem.dynamics import Intervention
from histem.system import CellSystem


def test_discrete_steps_stay_on_each_variables_levels(system: CellSystem) -> None:
    pop = system.sample_cells(200, "tiny")
    for slot in pop.schema.slots:
        if slot.levels is None:
            continue
        v = pop.values[slot.name]
        assert (v == v.round()).all()
        assert v.min() >= 0
        assert v.max() <= slot.levels - 1


def test_unknown_gene_rejected(system: CellSystem) -> None:
    with pytest.raises(KeyError, match="unknown genes"):
        system.sample_cells(
            2, "tiny", Intervention(name="Z", genes=("Z",), kind="activate")
        )
