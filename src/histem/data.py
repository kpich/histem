"""Datasets: observed cells plus the experimental conditions they came from."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad

from histem.dynamics import Intervention

# Sibling of the repo (src/histem/data.py -> ../../../data). Fetch scripts write here.
DATA_DIR = Path(__file__).resolve().parents[3] / "data"


@dataclass
class Dataset:
    name: str
    adata: ad.AnnData
    interventions: dict[
        str, Intervention
    ]  # condition label -> how to reproduce it in simulation
    condition_key: str = "condition"
    modality: str = "rna"

    @property
    def conditions(self) -> list[str]:
        present = set(self.adata.obs[self.condition_key].astype(str))
        return [c for c in self.interventions if c in present]

    def cells(self, condition: str) -> ad.AnnData:
        return self.adata[self.adata.obs[self.condition_key].astype(str) == condition]

    def split(self, held_out: list[str]) -> tuple[Dataset, Dataset]:
        """Split by condition, e.g. hold out perturbations to test generalisation."""
        train = {k: v for k, v in self.interventions.items() if k not in held_out}
        test = {k: v for k, v in self.interventions.items() if k in held_out}
        return (
            Dataset(
                f"{self.name}/train",
                self.adata,
                train,
                self.condition_key,
                self.modality,
            ),
            Dataset(
                f"{self.name}/test", self.adata, test, self.condition_key, self.modality
            ),
        )
