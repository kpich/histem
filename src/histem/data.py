"""Datasets: observed cells plus the experimental conditions they came from."""

from pathlib import Path
from typing import Self

import anndata as ad
from pydantic import model_validator

from histem.dynamics import Intervention
from histem.spec import FrozenSpec

# Sibling of the repo (src/histem/data.py -> ../../../data). Fetch scripts write here.
DATA_DIR = Path(__file__).resolve().parents[3] / "data"


class Dataset(FrozenSpec):
    name: str
    adata: ad.AnnData
    # condition label -> how to reproduce it in simulation
    interventions: dict[str, Intervention]
    condition_key: str = "condition"
    modality: str = "rna"

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.condition_key not in self.adata.obs:
            raise ValueError(f"adata.obs has no column {self.condition_key!r}")
        return self

    @property
    def conditions(self) -> list[str]:
        present = set(self.adata.obs[self.condition_key].astype(str))
        return [c for c in self.interventions if c in present]

    def cells(self, condition: str) -> ad.AnnData:
        return self.adata[self.adata.obs[self.condition_key].astype(str) == condition]

    def split(self, held_out: list[str]) -> tuple["Dataset", "Dataset"]:
        """Split by condition, e.g. hold out perturbations to test generalisation."""
        unknown = set(held_out) - set(self.interventions)
        if unknown:
            raise KeyError(f"unknown conditions: {sorted(unknown)}")
        ivs = self.interventions
        train = {k: v for k, v in ivs.items() if k not in held_out}
        test = {k: v for k, v in ivs.items() if k in held_out}
        return (
            self._renamed("train", train),
            self._renamed("test", test),
        )

    def _renamed(
        self, suffix: str, interventions: dict[str, Intervention]
    ) -> "Dataset":
        update = {"name": f"{self.name}/{suffix}", "interventions": interventions}
        return self.model_copy(update=update)
