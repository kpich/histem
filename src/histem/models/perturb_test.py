import pytest
import torch

from histem.dynamics import CONTROL, Inputs, Intervention
from histem.genes import GeneTable
from histem.models.perturb import Perturb
from histem.state import Slot, StateSchema

SLOT = Slot(name="genes", variables=("A", "B", "C"), levels=None)
SCHEMA = StateSchema(slots=(SLOT,))


def _perturb() -> Perturb:
    p = Perturb(slot=SLOT, gene_dim=4)
    p.bind_genes(GeneTable(genes=SLOT.variables, dim=4))
    return p


def _step(p: Perturb, iv: Intervention) -> torch.Tensor:
    pop = SCHEMA.empty(2)
    pop.values["genes"][:] = 0.5
    return p.step(pop, Inputs(torch.zeros(2, 0), iv))["genes"]


def test_targets_are_held_at_a_signed_magnitude_and_others_kept() -> None:
    p = _perturb()
    up = _step(p, Intervention(name="A_C", genes=("A", "C"), kind="activate"))
    assert up[:, [0, 2]].eq(2.0).all()
    assert up[:, 1].eq(0.5).all()
    down = _step(p, Intervention(name="B", genes=("B",), kind="knockout"))
    assert down[:, 1].eq(-4.0).all()
    assert _step(p, CONTROL).eq(0.5).all()


def test_magnitude_is_learned() -> None:
    p = _perturb()
    _step(p, Intervention(name="A", genes=("A",), kind="repress")).sum().backward()
    assert p.weights["base"].grad is not None
    assert p.weights["base"].grad.tolist() == [0.0, 2.0, 0.0]


def test_unknown_gene_rejected() -> None:
    with pytest.raises(KeyError, match="unknown genes"):
        _step(_perturb(), Intervention(name="Z", genes=("Z",), kind="activate"))
