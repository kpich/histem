"""Reference predictions for a perturbation, scored by the same energy distance as
`CountsDataset.loss`, in log-normalized space."""

import torch

from histem.data import CountsDataset
from histem.metrics import energy_distance


def control(
    ds: CountsDataset, condition: str, n: int, columns: list[int] | None = None
) -> torch.Tensor:
    """Predict that the perturbation does nothing. `columns` restricts the score
    to those features."""
    pred, real = ds.normalized("control", n), ds.normalized(condition, n)
    if columns is not None:
        pred, real = pred[:, columns], real[:, columns]
    return energy_distance(pred, real)


def additive_shift(ds: CountsDataset, condition: str) -> torch.Tensor:
    """Sum of each single perturbation's mean shift from control."""
    singles = ds.singles()
    genes = ds.interventions[condition].genes
    if missing := [g for g in genes if g not in singles]:
        raise KeyError(f"{condition}: no single perturbation for {missing}")
    base = ds.normalized("control").mean(dim=0)
    return torch.stack(
        [ds.normalized(singles[g]).mean(dim=0) - base for g in genes]
    ).sum(dim=0)


def additive(
    ds: CountsDataset, condition: str, n: int, columns: list[int] | None = None
) -> torch.Tensor:
    """Control cells shifted by the sum of the singles' mean shifts."""
    pred = ds.normalized("control", n) + additive_shift(ds, condition)
    real = ds.normalized(condition, n)
    if columns is not None:
        pred, real = pred[:, columns], real[:, columns]
    return energy_distance(pred, real)
