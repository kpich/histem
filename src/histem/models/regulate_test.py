import torch

from histem.dynamics import CONTROL, Inputs, Intervention
from histem.genes import GeneTable
from histem.mechanisms import Composite
from histem.models.perturb import Perturb
from histem.models.regulate import Regulate
from histem.state import Slot, StateSchema

SLOT = Slot(name="genes", variables=("A", "B", "C"), levels=None)
SCHEMA = StateSchema(slots=(SLOT,))


def _system() -> tuple[Composite, Regulate]:
    table = GeneTable(genes=SLOT.variables, dim=4)
    regulate = Regulate(slot=SLOT, gene_dim=4, rank=2)
    perturb = Perturb(slot=SLOT, gene_dim=4)
    for m in (regulate, perturb):
        m.bind_genes(table)
    return Composite(mechanisms={"regulate": regulate, "perturb": perturb}), regulate


def _run(c: Composite, iv: Intervention, steps: int = 3) -> torch.Tensor:
    pop = SCHEMA.empty(2)
    for _ in range(steps):
        pop = c.step(pop, Inputs(torch.zeros(2, 0), iv))
    return pop.values["genes"]


def test_control_stays_at_zero() -> None:
    c, regulate = _system()
    with torch.no_grad():
        regulate.weights["sink.bias"].fill_(1.0)
    assert _run(c, CONTROL).eq(0.0).all()


def test_perturbation_spreads_to_other_genes_and_targets_stay_held() -> None:
    c, regulate = _system()
    iv = Intervention(name="A", genes=("A",), kind="activate")
    assert _run(c, iv)[:, 1:].eq(0.0).all()  # sink starts at zero
    with torch.no_grad():
        regulate.weights["sink.bias"].fill_(1.0)
    out = _run(c, iv)
    assert out[:, 0].eq(2.0).all()
    assert out[:, 1:].ne(0.0).all()
    assert out.abs().le(3.0).all()


def test_coupling_is_learned() -> None:
    c, regulate = _system()
    with torch.no_grad():
        regulate.weights["sink.bias"].fill_(1.0)
    _run(c, Intervention(name="A", genes=("A",), kind="activate")).sum().backward()
    for name in ("source.weight", "sink.weight", "rate"):
        grad = regulate.weights[name].grad
        assert grad is not None
        assert grad.abs().sum() > 0
