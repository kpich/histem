import anndata as ad
import numpy as np
import pandas as pd
import torch

from histem import norman
from histem.suite import Suite


def _adata() -> ad.AnnData:
    perts = ["control"] * 4 + ["A"] * 3 + ["B"] * 3 + ["A_B"] * 3 + ["C_D"] * 2
    rng = np.random.default_rng(0)
    x = rng.poisson(3.0, (len(perts), 6)).astype(np.float32)
    x[:, 5] = 0  # constant, so never among the most variable
    obs = pd.DataFrame({"perturbation": perts}, index=[str(i) for i in range(len(x))])
    var = pd.DataFrame(index=["A", "B", "g1", "g2", "g3", "C"])
    return ad.AnnData(X=x, obs=obs, var=var)


def test_build_keeps_variable_and_target_genes_and_caps_cells() -> None:
    ds = norman.build(_adata(), n_genes=2, max_cells=3, seed=0)
    assert len(ds.features) == len(set(ds.features))
    assert {"A", "B", "C"} <= set(ds.features)
    assert ds.counts["control"].shape == (3, len(ds.features))
    assert ds.counts["C_D"].shape[0] == 2
    assert ds.interventions["A_B"].genes == ("A", "B")
    assert ds.interventions["control"].genes == ()
    assert ds.totals is None


def test_combo_split_only_holds_out_composable_doubles() -> None:
    ds = norman.build(_adata(), n_genes=2, max_cells=3, seed=0)
    assert norman.combo_split(ds, frac=1.0) == ["A_B"]


def test_system_trains_on_every_condition() -> None:
    torch.manual_seed(0)
    ds = norman.build(_adata(), n_genes=2, max_cells=3, seed=0)
    system = norman.make_system(
        ds, latents=3, levels=3, width=16, layers=1, burn_in=3, grad_steps=2
    )
    Suite(datasets=[ds], cells_per_condition=4).loss(system).backward()
    for p in system.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
