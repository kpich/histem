from pathlib import Path

import torch

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
        before = system.sample(20)
    with torch.no_grad(), seeded(0):
        after = loaded.sample(20)
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
