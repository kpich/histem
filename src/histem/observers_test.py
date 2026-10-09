import pytest
import torch
from pydantic import ValidationError

from histem.observers import NBCountObserver
from histem.state import Slot, StateSchema


def test_weight_shape_must_match_genes_and_drivers() -> None:
    with pytest.raises(ValidationError, match="weights shape"):
        NBCountObserver(
            genes=("g1", "g2"),
            drivers=("a",),
            weights=torch.zeros((3, 1)),
            bias=torch.zeros(2),
        )


def test_dispersion_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        NBCountObserver(
            genes=("g1",),
            drivers=("a",),
            weights=torch.zeros((1, 1)),
            bias=torch.zeros(1),
            dispersion=0,
        )


def test_nb_counts_have_requested_mean() -> None:
    schema = StateSchema(slots=(Slot(name="s", variables=("a",)),))
    obs = NBCountObserver(
        genes=("g",),
        drivers=("a",),
        weights=torch.zeros((1, 1)),
        bias=torch.tensor([2.0]),
        size_sd=0.0,
    )
    torch.manual_seed(0)
    counts = obs.observe(schema.empty(20_000))
    assert abs(counts.mean().item() - torch.e**2) < 0.2


def test_rsample_matches_nb_mean_and_is_differentiable() -> None:
    schema = StateSchema(slots=(Slot(name="s", variables=("a",)),))
    bias = torch.tensor([2.0], requires_grad=True)
    obs = NBCountObserver(
        genes=("g",),
        drivers=("a",),
        weights=torch.zeros((1, 1)),
        bias=bias,
        size_sd=0.0,
    )
    torch.manual_seed(0)
    counts = obs.rsample(schema.empty(20_000))
    assert abs(counts.mean().item() - torch.e**2) < 0.2
    counts.mean().backward()
    assert bias.grad is not None
    assert bias.grad.item() > 0
