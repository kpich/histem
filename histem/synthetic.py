"""A hand-written ground-truth world, used to check whether induction recovers known rules.

A toy myeloid/erythroid fork: a GATA1/PU1 toggle, chromatin gates on GATA1 and CEBPA,
a MYC <-> mitochondria loop, and a PU1-secreted signal (IL) that opens CEBPA chromatin.
This is not meant to be biologically accurate; it exercises every part of the framework.
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from histem.data import Dataset
from histem.dynamics import CONTROL, Intervention
from histem.models.logic import LogicDynamics
from histem.observers import NBCountObserver
from histem.simulator import Signaling
from histem.state import Slot, StateSchema
from histem.world import WorldModel

TFS = ("GATA1", "PU1", "FLI1", "KLF1", "CEBPA", "MYC")

SCHEMA = StateSchema((
    Slot("expr", TFS, levels=3, observed=True),
    Slot("chromatin", ("acc_GATA1", "acc_CEBPA"), levels=2),
    Slot("mito", ("mito_cn",), levels=3),
))

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


def make_observer(genes_per_driver: int = 4, housekeeping: int = 20, seed: int = 0) -> NBCountObserver:
    rng = np.random.default_rng(seed)
    genes, rows = [], []
    for j, d in enumerate(DRIVERS):
        for k in range(genes_per_driver):
            genes.append(f"{d}_t{k}")
            w = np.zeros(len(DRIVERS))
            w[j] = rng.uniform(1.5, 3.0)
            if rng.random() < 0.3:  # some targets are co-regulated
                w[rng.integers(len(DRIVERS))] += rng.uniform(-1.0, 1.0)
            rows.append(w)
    for k in range(housekeeping):
        genes.append(f"HK{k}")
        rows.append(np.zeros(len(DRIVERS)))
    weights = np.array(rows)
    bias = rng.uniform(-1.0, 1.5, len(genes))
    return NBCountObserver(tuple(genes), DRIVERS, weights, bias)


def make_world(program: str = PROGRAM) -> WorldModel:
    return WorldModel(
        dynamics=LogicDynamics.from_text(SCHEMA, program),
        observers={"rna": make_observer()},
        init=SCHEMA.uniform,
        burn_in=60,
        signaling=Signaling(autocrine=0.5, endocrine=0.5),
    )


def perturbations() -> dict[str, Intervention]:
    out = {"control": CONTROL}
    for tf in TFS:
        out[f"{tf}_KO"] = Intervention(f"{tf}_KO", ((tf, 0),))
        out[f"{tf}_OE"] = Intervention(f"{tf}_OE", ((tf, 2),))
    for a, b in [("GATA1", "PU1"), ("FLI1", "KLF1"), ("PU1", "CEBPA")]:
        out[f"{a}_KO+{b}_KO"] = Intervention(f"{a}_KO+{b}_KO", ((a, 0), (b, 0)))
    return out


def make_dataset(world: WorldModel | None = None, cells_per_condition: int = 300,
                 seed: int = 1) -> Dataset:
    world = world or make_world()
    rng = np.random.default_rng(seed)
    interventions = perturbations()
    parts = [world.sample(cells_per_condition, rng, iv) for iv in interventions.values()]
    adata = ad.concat(parts, index_unique="-")
    return Dataset("synthetic_fork", adata, interventions)
