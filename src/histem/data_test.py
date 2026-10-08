import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError

from histem import synthetic
from histem.data import CountsDataset, Dataset
from histem.dynamics import CONTROL, Intervention

IVS = {"control": CONTROL, "ko": Intervention(name="ko")}


def _adata() -> ad.AnnData:
    obs = pd.DataFrame({"condition": ["control", "ko", "ko"]}, index=["0", "1", "2"])
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.float32)
    return ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=["g1", "g2", "g3"]))


def test_from_anndata_groups_by_condition_and_keeps_full_totals() -> None:
    ds = CountsDataset.from_anndata("d", _adata(), IVS, features=["g1", "g3"])
    assert isinstance(ds, Dataset)
    assert ds.conditions == ["control", "ko"]
    assert ds.counts["ko"].tolist() == [[4, 6], [7, 9]]
    assert ds.totals is not None
    assert ds.totals["ko"].tolist() == [15, 24]


def test_missing_condition_column_rejected() -> None:
    with pytest.raises(ValueError, match="no column"):
        CountsDataset.from_anndata("d", _adata(), IVS, condition_key="nope")


def test_counts_must_match_features() -> None:
    with pytest.raises(ValidationError, match="features"):
        CountsDataset(
            name="d",
            features=("g1",),
            counts={"control": torch.zeros((2, 3))},
            interventions=IVS,
        )


def test_split_by_condition() -> None:
    ds = CountsDataset.from_anndata("d", _adata(), IVS)
    train, test = ds.split(["ko"])
    assert train.conditions == ["control"]
    assert test.conditions == ["ko"]
    assert test.counts["ko"].shape[0] == 2
    with pytest.raises(KeyError):
        ds.split(["missing"])


def test_loss_is_finite_scalar_on_a_batch() -> None:
    system = synthetic.make_system().model_copy(update={"burn_in": 2})
    ds = synthetic.make_dataset(system, cells_per_condition=20)
    loss = ds.loss(system, "control", 8)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
