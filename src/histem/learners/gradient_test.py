import torch

from histem.data import CountsDataset
from histem.learners.gradient import gradient_fit
from histem.suite import Suite
from histem.system import CellSystem


def test_updates_parameters_in_place(ds: CountsDataset, system: CellSystem) -> None:
    before = [p.detach().clone() for p in system.parameters()]
    seen: list[dict[str, float]] = []
    suite = Suite(datasets=[ds], cells_per_condition=8)
    losses = gradient_fit(system, suite, 2, callback=lambda *a: seen.append(a[2]))
    assert len(losses) == 2
    assert len(seen) == 2
    assert {"total", "genes/table", "observers/rna/log_dispersion"} <= seen[0].keys()
    after = list(system.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))
