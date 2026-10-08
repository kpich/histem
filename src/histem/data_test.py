import anndata as ad
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from histem.data import Dataset
from histem.dynamics import CONTROL, Intervention


def _adata() -> ad.AnnData:
    obs = pd.DataFrame({"condition": ["control", "ko", "ko"]}, index=["0", "1", "2"])
    return ad.AnnData(X=np.zeros((3, 2), np.float32), obs=obs)


def test_missing_condition_column_rejected() -> None:
    with pytest.raises(ValidationError, match="no column"):
        Dataset(name="d", adata=_adata(), interventions={}, condition_key="nope")


def test_split_by_condition() -> None:
    ivs = {"control": CONTROL, "ko": Intervention(name="ko")}
    ds = Dataset(name="d", adata=_adata(), interventions=ivs)
    train, test = ds.split(["ko"])
    assert train.conditions == ["control"]
    assert test.conditions == ["ko"]
    assert test.cells("ko").n_obs == 2
    with pytest.raises(KeyError):
        ds.split(["missing"])
