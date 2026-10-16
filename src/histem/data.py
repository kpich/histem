from collections.abc import Callable
from pathlib import Path
from typing import Protocol, Self, TypeVar, runtime_checkable

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from pydantic import model_validator

from histem.dynamics import Intervention
from histem.metrics import energy_distance, log_normalize
from histem.spec import FrozenSpec, Tensor
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
    # what the cells share, e.g. cell type and assay; split halves keep it
    context: str
    features: tuple[str, ...]
    counts: dict[str, Tensor]  # condition -> (n_cells, n_features)
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

    @property
    def conditions(self) -> list[str]:
        return [c for c in self.interventions if c in self.counts]

    def split(self, held_out: list[str]) -> tuple[Self, Self]:
        if unknown := set(held_out) - set(self.interventions):
            raise KeyError(f"unknown conditions: {sorted(unknown)}")
        keep = [c for c in self.interventions if c not in held_out]
        return self._subset("train", keep), self._subset("test", held_out)

    def _subset(self, suffix: str, conditions: list[str]) -> Self:
        return self.model_copy(
            update={
                "name": f"{self.name}/{suffix}",
                "counts": _pick(self.counts, conditions),
                "interventions": _pick(self.interventions, conditions),
            }
        )

    def singles(self) -> dict[str, str]:
        """Gene -> the condition that perturbs it alone."""
        return {
            iv.genes[0]: c for c, iv in self.interventions.items() if len(iv.genes) == 1
        }

    def loss(self, system: CellSystem, condition: str, n: int) -> torch.Tensor:
        observer = system.observers[self.modality]
        mine, theirs = _shared(self.features, observer.features)
        x = self.normalized(condition, n)[:, mine]
        y = self.simulated(system, condition, n)[:, theirs]
        x = x.to(y.device)
        return energy_distance(x, y)

    def normalized(self, condition: str, n: int | None = None) -> torch.Tensor:
        """Log-normalized observed cells; a random n of them if there are more.
        Library sizes are row sums over `features`, as for simulated cells."""
        obs = self.counts[condition]
        if n is not None and obs.shape[0] > n:
            obs = obs[torch.randperm(obs.shape[0])[:n]]
        return log_normalize(obs, obs.sum(dim=1))

    def simulated(self, system: CellSystem, condition: str, n: int) -> torch.Tensor:
        """n log-normalized simulated cells, over the observer's features."""
        sim = system.sample(
            n,
            self.context,
            self.interventions[condition],
            self.modality,
            reparam=torch.is_grad_enabled(),
        )
        return log_normalize(sim, sim.sum(dim=1))


def from_anndata(
    name: str,
    adata: ad.AnnData,
    intervention: Callable[[str], Intervention],
    *,
    n_genes: int,
    max_cells: int,
    seed: int,
    condition_key: str = "perturbation",
) -> CountsDataset:
    """The `n_genes` most variable genes in log-normalized expression, plus every
    measured target gene; at most `max_cells` cells per condition. `intervention`
    parses a condition label."""
    x = sp.csr_matrix(adata.X)
    labels = adata.obs[condition_key].astype(str).to_numpy()
    var = _log_norm_var(x)
    names = list(adata.var_names)
    cols = sorted(np.argsort(-var)[:n_genes].tolist())
    interventions = {c: intervention(c) for c in sorted(set(labels))}
    targets = {g for iv in interventions.values() for g in iv.genes}
    where = {g: i for i, g in enumerate(names)}
    cols += sorted(where[g] for g in targets if g in where and where[g] not in cols)

    sub = x[:, cols]
    rng = np.random.default_rng(seed)
    counts = {}
    for c in interventions:
        rows = np.flatnonzero(labels == c)
        if len(rows) > max_cells:
            rows = np.sort(rng.choice(rows, max_cells, replace=False))
        counts[c] = torch.as_tensor(sub[rows].toarray(), dtype=torch.float32)
    return CountsDataset(
        name=name,
        context=name,
        features=tuple(names[i] for i in cols),
        counts=counts,
        interventions=interventions,
    )


def _log_norm_var(x: sp.csr_matrix, chunk: int = 20_000) -> np.ndarray:
    """Per-gene variance of log CP10k, a block of rows at a time."""
    total = np.zeros(x.shape[1])
    square = np.zeros(x.shape[1])
    for start in range(0, x.shape[0], chunk):
        block = x[start : start + chunk]
        lib = np.maximum(np.asarray(block.sum(axis=1)).ravel(), 1)
        norm = sp.csr_matrix(block.multiply(1e4 / lib[:, None]))
        norm.data = np.log1p(norm.data)
        total += np.asarray(norm.sum(axis=0)).ravel()
        square += np.asarray(norm.multiply(norm).sum(axis=0)).ravel()
    mean = total / x.shape[0]
    var: np.ndarray = square / x.shape[0] - mean**2
    return var


def cached(key: str, build: Callable[[], CountsDataset]) -> CountsDataset:
    """`build()`, saved under ../data/processed/`key` on first use."""
    path = DATA_DIR / "processed" / key
    if (path / "config.json").exists():
        return CountsDataset.load(path)
    ds = build()
    ds.save(path)
    return ds


def combo_split(ds: CountsDataset, frac: float = 0.2, seed: int = 0) -> list[str]:
    """A random `frac` of the double perturbations whose singles are both present."""
    singles = ds.singles()
    combos = [
        c
        for c, iv in ds.interventions.items()
        if len(iv.genes) == 2 and set(iv.genes) <= singles.keys()
    ]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(combos), generator=g)[: round(frac * len(combos))]
    return sorted(combos[i] for i in idx.tolist())


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
