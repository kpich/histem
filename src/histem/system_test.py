from pathlib import Path

import torch

from histem.data import CountsDataset
from histem.latent import make_system
from histem.mechanisms import Composite
from histem.models.diffusion import DiffusionKernel
from histem.observers import DecoderObserver
from histem.rng import seeded
from histem.system import CellSystem, UniformInit


def test_save_load_round_trip(tmp_path: Path, system: CellSystem) -> None:
    system.save(tmp_path)
    loaded = CellSystem.load(tmp_path)
    assert isinstance(loaded.dynamics, Composite)
    assert isinstance(loaded.dynamics.mechanisms["latent"], DiffusionKernel)
    assert isinstance(loaded.observers["rna"], DecoderObserver)
    assert isinstance(loaded.init, UniformInit)
    with torch.no_grad(), seeded(0):
        before = system.sample(20, "tiny")
    with torch.no_grad(), seeded(0):
        after = loaded.sample(20, "tiny")
    assert torch.equal(before, after)


def _tied(system: CellSystem) -> bool:
    table = system.genes
    assert isinstance(system.dynamics, Composite)
    parts = [*system.dynamics.mechanisms.values(), system.observers["rna"]]
    return table is not None and all(p._genes is table for p in parts)  # type: ignore[attr-defined]


def test_gene_table_is_saved_once_and_stays_tied(
    tmp_path: Path, system: CellSystem
) -> None:
    system.save(tmp_path)
    loaded = CellSystem.load(tmp_path)
    assert _tied(loaded)
    assert _tied(loaded.to("cpu"))
    tensors = torch.load(tmp_path / "tensors.pt", weights_only=True)
    assert system.genes is not None
    table = system.genes.weights["table"]
    assert sum(t.shape == table.shape for t in tensors.values()) == 1


def test_load_weights_into_a_system_with_more_genes_and_contexts(
    ds: CountsDataset, system: CellSystem
) -> None:
    other = ds.model_copy(
        update={
            "name": "other",
            "context": "other",
            "features": ("A", "B", "g1", "new"),
        }
    )
    bigger = make_system(
        [ds, other], latents=3, levels=3, width=16, layers=1, steps=3, grad_steps=2
    )
    fresh = bigger.load_weights(system)
    assert sorted(fresh) == [
        "dynamics/latent/context.other",
        "observers/rna/bias.other",
    ]
    assert system.genes is not None
    assert bigger.genes is not None
    old, new = system.genes, bigger.genes
    assert torch.equal(new.embed(old.genes), old.embed(old.genes))
    assert not new.embed(("new",)).any()
    means = []
    for s in (system, bigger):
        obs = s.observers["rna"]
        assert isinstance(obs, DecoderObserver)
        with torch.no_grad(), seeded(0):
            means.append(obs.log_mean(s.sample_cells(20, "tiny"), "tiny"))
    assert torch.allclose(means[0], means[1][:, :4])


def test_one_kernel_context_and_decoder_bias_per_dataset(
    ds: CountsDataset,
) -> None:
    other = ds.model_copy(update={"name": "other", "context": "other"})
    system = make_system([ds, other], latents=3, levels=3, width=16, layers=1)
    names = dict(system.named_parameters())
    for c in ("tiny", "other"):
        assert f"dynamics/latent/context.{c}" in names
        assert f"observers/rna/bias.{c}" in names
