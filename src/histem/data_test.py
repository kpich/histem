import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError

from histem.data import CountsDataset, Dataset, combo_split, from_anndata, single_split
from histem.dynamics import CONTROL, Intervention
from histem.latent import make_system
from histem.system import CellSystem

IVS = {"control": CONTROL, "ko": Intervention(name="ko")}


def _parse(label: str) -> Intervention:
    if label == "control":
        return CONTROL
    return Intervention(name=label, genes=tuple(label.split("_")), kind="activate")


def _adata() -> ad.AnnData:
    perts = ["control"] * 4 + ["A"] * 3 + ["B"] * 3 + ["A_B"] * 3 + ["C_D"] * 2
    rng = np.random.default_rng(0)
    x = rng.poisson(3.0, (len(perts), 6)).astype(np.float32)
    x[:, 5] = 0  # constant, so never among the most variable
    obs = pd.DataFrame({"perturbation": perts}, index=[str(i) for i in range(len(x))])
    var = pd.DataFrame(index=["A", "B", "g1", "g2", "g3", "C"])
    return ad.AnnData(X=x, obs=obs, var=var)


def _built() -> CountsDataset:
    return from_anndata("d", _adata(), _parse, n_genes=2, max_cells=3, seed=0)


def test_from_anndata_keeps_variable_and_target_genes_and_caps_cells() -> None:
    ds = _built()
    assert isinstance(ds, Dataset)
    assert len(ds.features) == len(set(ds.features))
    assert {"A", "B", "C"} <= set(ds.features)
    assert ds.counts["control"].shape == (3, len(ds.features))
    assert ds.counts["C_D"].shape[0] == 2
    assert ds.interventions["A_B"].genes == ("A", "B")


def test_combo_split_only_holds_out_composable_doubles() -> None:
    assert combo_split(_built(), frac=1.0) == ["A_B"]


def test_single_split_only_holds_out_singles() -> None:
    assert single_split(_built(), frac=1.0) == ["A", "B"]
    assert single_split(_built(), frac=0.5, seed=1) in (["A"], ["B"])


def test_counts_must_match_features() -> None:
    with pytest.raises(ValidationError, match="features"):
        CountsDataset(
            name="d",
            context="d",
            features=("g1",),
            counts={"control": torch.zeros((2, 3))},
            interventions=IVS,
        )


def test_split_by_condition() -> None:
    ds = CountsDataset(
        name="d",
        context="d",
        features=("g1",),
        counts={"control": torch.zeros((2, 1)), "ko": torch.zeros((3, 1))},
        interventions=IVS,
    )
    train, test = ds.split(["ko"])
    assert train.conditions == ["control"]
    assert test.conditions == ["ko"]
    assert test.counts["ko"].shape[0] == 3
    with pytest.raises(KeyError):
        ds.split(["missing"])


def test_loss_is_finite_scalar_on_a_batch(
    ds: CountsDataset, system: CellSystem
) -> None:
    loss = ds.loss(system, "control", 8)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_simulated_cells_cover_only_the_datasets_features(ds: CountsDataset) -> None:
    other = ds.model_copy(
        update={
            "name": "other",
            "context": "other",
            "features": ("g3", "g1"),
            "counts": {c: x[:, :2] for c, x in ds.counts.items()},
        }
    )
    system = make_system([ds, other], latents=2, levels=2, width=8, layers=1, steps=1)
    with torch.no_grad():
        sim = other.simulated(system, "A", 5)
    assert sim.shape == (5, 2)
    assert torch.allclose(sim.expm1().sum(dim=1), torch.full((5,), 1e4), rtol=1e-3)
