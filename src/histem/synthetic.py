"""Toy ground truth: GATA1/PU1 fork, chromatin gates, MYC/mito loop, IL signal."""

import torch

from histem.data import CountsDataset
from histem.dynamics import CONTROL, Intervention
from histem.models.logic import LogicDynamics
from histem.models.neural import NeuralDynamics
from histem.observers import NBCountObserver
from histem.rng import seeded
from histem.simulator import Signaling
from histem.state import Slot, StateSchema
from histem.system import CellSystem

TFS = ("GATA1", "PU1", "FLI1", "KLF1", "CEBPA", "MYC")

SCHEMA = StateSchema(
    slots=(
        Slot(name="expr", variables=TFS, levels=3, observed=True),
        Slot(name="chromatin", variables=("acc_GATA1", "acc_CEBPA"), levels=2),
        Slot(name="mito", variables=("mito_cn",), levels=3),
    )
)

PROGRAM = """
GATA1 <- 2 if acc_GATA1 >= 1 and (GATA1 >= 1 or FLI1 >= 1) and not PU1 >= 2 else 0
PU1 <- 2 if (PU1 >= 1 or CEBPA >= 1) and not GATA1 >= 2 else 0
FLI1 <- 2 if GATA1 >= 1 and not KLF1 >= 1 else 0
KLF1 <- 2 if GATA1 >= 2 and not FLI1 >= 2 else 0
CEBPA <- 2 if acc_CEBPA >= 1 and PU1 >= 1 else 0
MYC <- 2 if mito_cn >= 1 and not (KLF1 >= 2 or CEBPA >= 2) else 1
mito_cn <- 2 if MYC >= 2 else 1
acc_GATA1 <- not CEBPA >= 2
acc_CEBPA <- in_IL >= 0.5 or acc_CEBPA >= 1
emit IL <- 1.0 if PU1 >= 2 else 0.0
rate acc_GATA1 = 0.05
rate acc_CEBPA = 0.05
rate mito_cn = 0.1
"""

DRIVERS = (*TFS, "mito_cn")


def make_observer(
    genes_per_driver: int = 4, housekeeping: int = 20, seed: int = 0
) -> NBCountObserver:
    g = torch.Generator().manual_seed(seed)

    def uniform(lo: float, hi: float, n: int = 1) -> torch.Tensor:
        return lo + (hi - lo) * torch.rand(n, generator=g)

    genes, rows = [], []
    for j, d in enumerate(DRIVERS):
        for k in range(genes_per_driver):
            genes.append(f"{d}_t{k}")
            w = torch.zeros(len(DRIVERS))
            w[j] = uniform(1.5, 3.0)
            if torch.rand(1, generator=g) < 0.3:
                w[torch.randint(len(DRIVERS), (1,), generator=g)] += uniform(-1.0, 1.0)
            rows.append(w)
    for k in range(housekeeping):
        genes.append(f"HK{k}")
        rows.append(torch.zeros(len(DRIVERS)))
    return NBCountObserver(
        genes=tuple(genes),
        drivers=DRIVERS,
        weights=torch.stack(rows),
        bias=uniform(-1.0, 1.5, len(genes)),
    )


def make_system(program: str = PROGRAM) -> CellSystem:
    return CellSystem(
        dynamics=LogicDynamics.from_text(SCHEMA, program),
        observers={"rna": make_observer()},
        burn_in=60,
        signaling=Signaling(autocrine=0.5, endocrine=0.5),
    )


# continuous drivers in level units (so clamps mean the same thing) plus latents
NEURAL_SCHEMA = StateSchema(
    slots=(
        Slot(name="drivers", variables=DRIVERS, kind="continuous"),
        Slot(
            name="latent", variables=tuple(f"h{i}" for i in range(4)), kind="continuous"
        ),
    )
)


def make_neural_system(hidden: int = 64, device: str = "cpu") -> CellSystem:
    """Learnable dynamics under the true observer, rescaled from [0, 1] features to
    level units."""
    obs = make_observer()
    return CellSystem(
        dynamics=NeuralDynamics(
            state_schema=NEURAL_SCHEMA, signals=("IL",), hidden=hidden
        ),
        observers={"rna": obs.model_copy(update={"weights": obs.weights / 2})},
        burn_in=30,
        signaling=Signaling(autocrine=0.5, endocrine=0.5),
    ).to(device)


def perturbations() -> dict[str, Intervention]:
    out = {"control": CONTROL}
    for tf in TFS:
        out[f"{tf}_KO"] = Intervention(name=f"{tf}_KO", clamps=((tf, 0),))
        out[f"{tf}_OE"] = Intervention(name=f"{tf}_OE", clamps=((tf, 2),))
    for a, b in [("GATA1", "PU1"), ("FLI1", "KLF1"), ("PU1", "CEBPA")]:
        name = f"{a}_KO+{b}_KO"
        out[name] = Intervention(name=name, clamps=((a, 0), (b, 0)))
    return out


def make_dataset(
    system: CellSystem | None = None, cells_per_condition: int = 300, seed: int = 1
) -> CountsDataset:
    system = system or make_system()
    interventions = perturbations()
    with torch.no_grad(), seeded(seed):
        counts = {
            name: system.sample(cells_per_condition, iv)
            for name, iv in interventions.items()
        }
    return CountsDataset(
        name="synthetic_fork",
        features=system.observers["rna"].features,
        counts=counts,
        interventions=interventions,
    )
