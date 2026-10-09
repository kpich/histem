import pytest
import torch

from histem import synthetic
from histem.learners.gradient import gradient_fit
from histem.suite import Suite


def _suite() -> Suite:
    ds = synthetic.make_dataset(synthetic.make_system(), cells_per_condition=10)
    _, control = ds.split(["control"])
    return Suite(datasets=[control], cells_per_condition=10)


def test_updates_parameters_in_place() -> None:
    torch.manual_seed(0)
    system = synthetic.make_neural_system(hidden=8).model_copy(update={"burn_in": 2})
    before = [p.detach().clone() for p in system.parameters()]
    log = gradient_fit(system, _suite(), 2)
    assert len(log.losses) == 2
    after = list(system.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_rejects_systems_without_parameters() -> None:
    with pytest.raises(ValueError, match="no learnable"):
        gradient_fit(synthetic.make_system(), _suite(), 1)
