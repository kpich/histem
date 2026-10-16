"""Replogle et al. 2022: CRISPRi knockdown of essential genes in K562."""

import anndata as ad

from histem.data import DATA_DIR, CountsDataset, cached, from_anndata
from histem.dynamics import CONTROL, Intervention

RAW = (
    DATA_DIR
    / "raw"
    / "replogle_k562_essential"
    / "ReplogleWeissman2022_K562_essential.h5ad"
)


def intervention(label: str) -> Intervention:
    """ "A" represses A."""
    if label == "control":
        return CONTROL
    return Intervention(name=label, genes=(label,), kind="repress")


def load(n_genes: int = 1000, max_cells: int = 500, seed: int = 0) -> CountsDataset:
    return cached(
        f"replogle_k562_essential_g{n_genes}_c{max_cells}_s{seed}",
        lambda: from_anndata(
            "replogle_k562_essential",
            ad.read_h5ad(RAW),
            intervention,
            n_genes=n_genes,
            max_cells=max_cells,
            seed=seed,
        ),
    )
