"""Norman et al. 2019: CRISPRa single and double perturbations in K562."""

import anndata as ad

from histem.data import DATA_DIR, CountsDataset, cached, from_anndata
from histem.dynamics import CONTROL, Intervention

RAW = DATA_DIR / "raw" / "norman2019" / "NormanWeissman2019_filtered.h5ad"


def intervention(label: str) -> Intervention:
    """ "A_B" activates A and B."""
    if label == "control":
        return CONTROL
    return Intervention(name=label, genes=tuple(label.split("_")), kind="activate")


def load(n_genes: int = 1000, max_cells: int = 500, seed: int = 0) -> CountsDataset:
    return cached(
        f"norman2019_g{n_genes}_c{max_cells}_s{seed}",
        lambda: from_anndata(
            "norman2019",
            ad.read_h5ad(RAW),
            intervention,
            n_genes=n_genes,
            max_cells=max_cells,
            seed=seed,
        ),
    )
