import torch

from histem.genes import GeneTable
from histem.observers import DecoderObserver
from histem.state import Slot, StateSchema, latent_schema

SCHEMA = latent_schema(2, 3)


def _observer() -> DecoderObserver:
    obs = DecoderObserver(
        genes=("g",),
        drivers=tuple(SCHEMA.variables),
        contexts=("c",),
        gene_dim=4,
        size_sd=0.0,
    )
    obs.bind_genes(GeneTable(genes=("g",), dim=4))
    obs.set_bias("c", ("g",), torch.tensor([2.0]))
    return obs


def test_counts_start_at_the_bias_mean() -> None:
    torch.manual_seed(0)
    counts = _observer().observe(SCHEMA.uniform(20_000), "c")
    assert abs(counts.mean().item() - torch.e**2) < 0.2


def test_rsample_matches_the_mean_and_is_differentiable() -> None:
    obs = _observer()
    torch.manual_seed(0)
    counts = obs.rsample(SCHEMA.uniform(20_000), "c")
    assert abs(counts.mean().item() - torch.e**2) < 0.2
    counts.mean().backward()
    bias = obs.weights["bias.c"]
    assert bias.grad is not None
    assert bias.grad.item() > 0


def test_activity_adds_to_its_genes_log_mean() -> None:
    schema = StateSchema(
        slots=(*SCHEMA.slots, Slot(name="act", variables=("x", "g"), levels=None))
    )
    obs, active = (
        DecoderObserver(
            genes=("g", "h"),
            drivers=tuple(SCHEMA.variables),
            contexts=("c",),
            activity=activity,
            gene_dim=4,
            size_sd=0.0,
        )
        for activity in (None, "act")
    )
    for o in (obs, active):
        o.bind_genes(GeneTable(genes=("g", "h"), dim=4))
    pop = schema.empty(3)
    pop.values["act"][:] = torch.tensor([5.0, 1.0])
    # W starts at zero, so without activity both are the zero bias
    diff = active.log_mean(pop, "c") - obs.log_mean(pop, "c")
    assert diff.tolist() == [[1.0, 0.0]] * 3
