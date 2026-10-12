"""Reference predictions for a perturbation, scored by the same energy distance as
`CountsDataset.loss`, in log-normalized space."""

import torch

from histem.data import CountsDataset
from histem.metrics import energy_distance


def control(ds: CountsDataset, condition: str, n: int) -> torch.Tensor:
    """Predict that the perturbation does nothing."""
    return energy_distance(ds.normalized("control", n), ds.normalized(condition, n))


def additive_shift(ds: CountsDataset, condition: str) -> torch.Tensor:
    """Sum of each single perturbation's mean shift from control."""
    singles = {
        iv.genes[0]: c for c, iv in ds.interventions.items() if len(iv.genes) == 1
    }
    genes = ds.interventions[condition].genes
    if missing := [g for g in genes if g not in singles]:
        raise KeyError(f"{condition}: no single perturbation for {missing}")
    base = ds.normalized("control").mean(dim=0)
    return torch.stack(
        [ds.normalized(singles[g]).mean(dim=0) - base for g in genes]
    ).sum(dim=0)


def additive(ds: CountsDataset, condition: str, n: int) -> torch.Tensor:
    """Control cells shifted by the sum of the singles' mean shifts."""
    pred = ds.normalized("control", n) + additive_shift(ds, condition)
    return energy_distance(pred, ds.normalized(condition, n))
