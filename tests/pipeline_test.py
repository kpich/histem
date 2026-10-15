import math

import anndata as ad
import numpy as np
import pandas as pd

from histem import baselines, norman
from histem.data import combo_split, from_anndata
from histem.latent import make_system
from histem.learners.gradient import gradient_fit
from histem.suite import Suite


def _adata() -> ad.AnnData:
    perts = ["control"] * 8 + ["A"] * 6 + ["B"] * 6 + ["A_B"] * 6
    rng = np.random.default_rng(0)
    x = rng.poisson(3.0, (len(perts), 8)).astype(np.float32)
    obs = pd.DataFrame({"perturbation": perts}, index=[str(i) for i in range(len(x))])
    var = pd.DataFrame(index=["A", "B", *(f"g{i}" for i in range(6))])
    return ad.AnnData(X=x, obs=obs, var=var)


def test_fit_on_singles_and_score_a_held_out_double() -> None:
    ds = from_anndata(
        "d", _adata(), norman.intervention, n_genes=4, max_cells=5, seed=0
    )
    held_out = combo_split(ds, frac=1.0)
    assert held_out == ["A_B"]
    train, test = ds.split(held_out)
    system = make_system([ds], latents=3, levels=3, width=16, layers=1, steps=3)
    gradient_fit(system, Suite(datasets=[train], cells_per_condition=5), 2)

    scores = Suite(datasets=[train, test], cells_per_condition=5).evaluate(system)
    assert all(math.isfinite(v) for v in scores.fit.values())
    assert math.isfinite(float(baselines.additive(ds, "A_B", 5)))
