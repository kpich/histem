from pathlib import Path

import torch

from histem.models.diffusion import DiffusionKernel
from histem.observers import DecoderObserver
from histem.rng import seeded
from histem.system import CellSystem, UniformInit


def test_save_load_round_trip(tmp_path: Path, system: CellSystem) -> None:
    system.save(tmp_path)
    loaded = CellSystem.load(tmp_path)
    assert isinstance(loaded.dynamics, DiffusionKernel)
    assert isinstance(loaded.observers["rna"], DecoderObserver)
    assert isinstance(loaded.init, UniformInit)
    with torch.no_grad(), seeded(0):
        before = system.sample(20)
    with torch.no_grad(), seeded(0):
        after = loaded.sample(20)
    assert torch.equal(before, after)
