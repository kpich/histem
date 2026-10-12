import torch

from histem.data import CountsDataset
from histem.suite import Suite
from histem.system import CellSystem


def test_system_trains_on_every_condition(
    ds: CountsDataset, system: CellSystem
) -> None:
    Suite(datasets=[ds], cells_per_condition=4).loss(system).backward()
    for p in system.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
