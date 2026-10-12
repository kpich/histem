import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError

from histem import synthetic
from histem.data import CountsDataset, Dataset, combo_split, from_anndata
from histem.dynamics import CONTROL, Intervention

IVS = {"control": CONTROL, "ko": Intervention(name="ko")}


def _parse(label: str) -> Intervention:
    if label == "control":
        return CONTROL
    return Intervention(name=label, genes=tuple(label.split("_")))


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


def test_counts_must_match_features() -> None:
    with pytest.raises(ValidationError, match="features"):
        CountsDataset(
            name="d",
            features=("g1",),
            counts={"control": torch.zeros((2, 3))},
            interventions=IVS,
        )


def test_split_by_condition() -> None:
    ds = CountsDataset(
        name="d",
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


def test_loss_is_finite_scalar_on_a_batch() -> None:
    system = synthetic.make_system().model_copy(update={"burn_in": 2})
    ds = synthetic.make_dataset(system, cells_per_condition=20)
    loss = ds.loss(system, "control", 8)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
