from pathlib import Path
from typing import Protocol, Self, TypeVar, runtime_checkable

import anndata as ad
import numpy as np
import torch
from pydantic import model_validator

from histem.dynamics import Intervention
from histem.metrics import energy_distance, log_normalize
from histem.spec import FrozenSpec
from histem.system import CellSystem

# ../data next to the repo
DATA_DIR = Path(__file__).resolve().parents[3] / "data"

T = TypeVar("T")


@runtime_checkable
class Dataset(Protocol):
    """Observed data for a set of conditions, plus how to score a system against it."""

    @property
    def name(self) -> str: ...

    @property
    def conditions(self) -> list[str]: ...

    def loss(self, system: CellSystem, condition: str, n: int) -> torch.Tensor:
        """Scalar loss on a batch of n cells; differentiable if the system is."""
        ...


class CountsDataset(FrozenSpec):
    """Per-condition single-cell counts, scored by energy distance on log-normalized
    shared features."""

    name: str
    features: tuple[str, ...]
    counts: dict[str, torch.Tensor]  # condition -> (n_cells, n_features)
    # condition -> (n_cells,) library size; defaults to the row sums of `counts`
    totals: dict[str, torch.Tensor] | None = None
    interventions: dict[str, Intervention]
    modality: str = "rna"

    @model_validator(mode="after")
    def _check(self) -> Self:
        if unknown := set(self.counts) - set(self.interventions):
            raise ValueError(f"counts for unknown conditions: {sorted(unknown)}")
        for cond, x in self.counts.items():
            if x.ndim != 2 or x.shape[1] != len(self.features):
                raise ValueError(
                    f"{cond}: counts shape {tuple(x.shape)} doesn't match "
                    f"{len(self.features)} features"
                )
        return self

    @classmethod
    def from_anndata(
        cls,
        name: str,
        adata: ad.AnnData,
        interventions: dict[str, Intervention],
        *,
        condition_key: str = "condition",
        features: list[str] | None = None,
        modality: str = "rna",
    ) -> Self:
        if condition_key not in adata.obs:
            raise ValueError(f"adata.obs has no column {condition_key!r}")
        features = features or list(adata.var_names)
        labels = adata.obs[condition_key].astype(str).to_numpy()
        counts, totals = {}, {}
        for cond in interventions:
            sub = adata[labels == cond]
            if sub.n_obs == 0:
                continue
            totals[cond] = torch.as_tensor(_dense(sub.X).sum(axis=1))
            counts[cond] = torch.as_tensor(_dense(sub[:, features].X))
        return cls(
            name=name,
            features=tuple(features),
            counts=counts,
            totals=totals,
            interventions=interventions,
            modality=modality,
        )

    @property
    def conditions(self) -> list[str]:
        return [c for c in self.interventions if c in self.counts]

    def split(self, held_out: list[str]) -> tuple[Self, Self]:
        if unknown := set(held_out) - set(self.interventions):
            raise KeyError(f"unknown conditions: {sorted(unknown)}")
        keep = [c for c in self.interventions if c not in held_out]
        return self._subset("train", keep), self._subset("test", held_out)

    def _subset(self, suffix: str, conditions: list[str]) -> Self:
        totals = None if self.totals is None else _pick(self.totals, conditions)
        return self.model_copy(
            update={
                "name": f"{self.name}/{suffix}",
                "counts": _pick(self.counts, conditions),
                "totals": totals,
                "interventions": _pick(self.interventions, conditions),
            }
        )

    def loss(self, system: CellSystem, condition: str, n: int) -> torch.Tensor:
        observer = system.observers[self.modality]
        mine, theirs = _shared(self.features, observer.features)
        obs = self.counts[condition]
        tot = obs.sum(dim=1) if self.totals is None else self.totals[condition]
        if obs.shape[0] > n:
            idx = torch.randperm(obs.shape[0])[:n]
            obs, tot = obs[idx], tot[idx]
        sim = system.sample(n, self.interventions[condition], self.modality)
        x = log_normalize(obs[:, mine], tot).to(sim.device)
        y = log_normalize(sim[:, theirs], sim.sum(dim=1))
        return energy_distance(x, y)


def _shared(
    ours: tuple[str, ...], theirs: tuple[str, ...]
) -> tuple[list[int], list[int]]:
    where = {f: i for i, f in enumerate(theirs)}
    pairs = [(i, where[f]) for i, f in enumerate(ours) if f in where]
    if not pairs:
        raise ValueError("observed and simulated data share no features")
    return [i for i, _ in pairs], [j for _, j in pairs]


def _pick(d: dict[str, T], keys: list[str]) -> dict[str, T]:
    return {k: v for k, v in d.items() if k in keys}


def _dense(x: object) -> np.ndarray:
    dense = x.toarray() if hasattr(x, "toarray") else x
    return np.asarray(dense, dtype=np.float32)
