import torch

from histem.observers import DecoderObserver
from histem.state import latent_schema

SCHEMA = latent_schema(2, 3)


def _observer() -> DecoderObserver:
    obs = DecoderObserver(genes=("g",), drivers=tuple(SCHEMA.variables), size_sd=0.0)
    obs.set_bias(torch.tensor([2.0]))
    return obs


def test_counts_start_at_the_bias_mean() -> None:
    torch.manual_seed(0)
    counts = _observer().observe(SCHEMA.uniform(20_000))
    assert abs(counts.mean().item() - torch.e**2) < 0.2


def test_rsample_matches_the_mean_and_is_differentiable() -> None:
    obs = _observer()
    torch.manual_seed(0)
    counts = obs.rsample(SCHEMA.uniform(20_000))
    assert abs(counts.mean().item() - torch.e**2) < 0.2
    counts.mean().backward()
    bias = obs.weights["bias"]
    assert bias.grad is not None
    assert bias.grad.item() > 0
