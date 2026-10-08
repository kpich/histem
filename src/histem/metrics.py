import anndata as ad
import numpy as np
from scipy.spatial.distance import cdist


def log_normalize(
    adata: ad.AnnData, genes: list[str], target: float = 1e4
) -> np.ndarray:
    x = adata[:, genes].X
    x = np.asarray(x.todense() if hasattr(x, "todense") else x, dtype=np.float64)
    total = adata.X.sum(axis=1)
    total = np.asarray(total).reshape(-1, 1).clip(min=1)
    out: np.ndarray = np.log1p(x / total * target)
    return out


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Energy distance (Székely); zero iff the distributions match."""
    return float(2 * cdist(x, y).mean() - cdist(x, x).mean() - cdist(y, y).mean())


def population_distance(
    observed: ad.AnnData,
    simulated: ad.AnnData,
    max_cells: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    genes = [g for g in observed.var_names if g in set(simulated.var_names)]
    if not genes:
        raise ValueError("observed and simulated data share no genes")
    rng = rng or np.random.default_rng(0)

    def take(a: ad.AnnData) -> ad.AnnData:
        return (
            a
            if a.n_obs <= max_cells
            else a[rng.choice(a.n_obs, max_cells, replace=False)]
        )

    return energy_distance(
        log_normalize(take(observed), genes), log_normalize(take(simulated), genes)
    )
