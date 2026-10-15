import pytest
import torch

from histem.data import CountsDataset
from histem.dynamics import CONTROL, Intervention
from histem.latent import make_system
from histem.system import CellSystem


@pytest.fixture
def ds() -> CountsDataset:
    """Genes A and B each raise their own counts; A_B raises both."""
    torch.manual_seed(0)
    base = torch.full((20, 4), 10.0)
    scale = {
        "control": [1.0, 1, 1, 1],
        "A": [4.0, 1, 1, 1],
        "B": [1.0, 4, 1, 1],
        "A_B": [4.0, 4, 1, 1],
    }
    return CountsDataset(
        name="tiny",
        context="tiny",
        features=("A", "B", "g1", "g2"),
        counts={c: torch.poisson(base * torch.tensor(s)) for c, s in scale.items()},
        interventions={
            "control": CONTROL,
            "A": Intervention(name="A", genes=("A",), kind="activate"),
            "B": Intervention(name="B", genes=("B",), kind="activate"),
            "A_B": Intervention(name="A_B", genes=("A", "B"), kind="activate"),
        },
    )


@pytest.fixture
def system(ds: CountsDataset) -> CellSystem:
    torch.manual_seed(0)
    return make_system(
        [ds], latents=3, levels=3, width=16, layers=1, steps=3, grad_steps=2
    )
