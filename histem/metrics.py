"""Distances between observed and simulated cell populations.

Snapshot data has no cell-to-cell pairing between experiment and simulation, so every
fit term compares distributions.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
from scipy.spatial.distance import cdist


def log_normalize(adata: ad.AnnData, genes: list[str], target: float = 1e4) -> np.ndarray:
    X = adata[:, genes].X
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)
    total = adata.X.sum(axis=1)
    total = np.asarray(total).reshape(-1, 1).clip(min=1)
    return np.log1p(X / total * target)


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Energy distance (Székely). Zero iff the distributions match; no bandwidth to tune."""
    return float(
        2 * cdist(x, y).mean() - cdist(x, x).mean() - cdist(y, y).mean()
    )


def population_distance(observed: ad.AnnData, simulated: ad.AnnData, max_cells: int = 1000,
                        rng: np.random.Generator | None = None) -> float:
    genes = [g for g in observed.var_names if g in set(simulated.var_names)]
    if not genes:
        raise ValueError("observed and simulated data share no genes")
    rng = rng or np.random.default_rng(0)

    def take(a):
        return a if a.n_obs <= max_cells else a[rng.choice(a.n_obs, max_cells, replace=False)]

    return energy_distance(log_normalize(take(observed), genes), log_normalize(take(simulated), genes))
