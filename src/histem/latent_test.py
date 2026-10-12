import torch

from histem.data import CountsDataset
from histem.dynamics import CONTROL, Intervention
from histem.latent import make_system
from histem.suite import Suite


def test_system_trains_on_every_condition() -> None:
    torch.manual_seed(0)
    ds = CountsDataset(
        name="d",
        features=("A", "g1", "g2"),
        counts={c: torch.poisson(torch.full((5, 3), 3.0)) for c in ["control", "A"]},
        interventions={"control": CONTROL, "A": Intervention(name="A", genes=("A",))},
    )
    system = make_system(
        ds, latents=3, levels=3, width=16, layers=1, burn_in=3, grad_steps=2
    )
    Suite(datasets=[ds], cells_per_condition=4).loss(system).backward()
    for p in system.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
