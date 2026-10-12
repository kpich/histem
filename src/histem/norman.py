"""Norman et al. 2019: CRISPRa single and double perturbations in K562."""

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from histem.data import DATA_DIR, CountsDataset
from histem.dynamics import CONTROL, Intervention
from histem.models.diffusion import DiffusionKernel
from histem.observers import DecoderObserver
from histem.state import latent_schema
from histem.system import CellSystem

RAW = DATA_DIR / "raw" / "norman2019" / "NormanWeissman2019_filtered.h5ad"


def load(n_genes: int = 1000, max_cells: int = 500, seed: int = 0) -> CountsDataset:
    """Cached under ../data/processed."""
    cache = DATA_DIR / "processed" / f"norman2019_g{n_genes}_c{max_cells}_s{seed}"
    if (cache / "config.json").exists():
        return CountsDataset.load(cache)
    ds = build(ad.read_h5ad(RAW), n_genes, max_cells, seed)
    ds.save(cache)
    return ds


def intervention(perturbation: str) -> Intervention:
    if perturbation == "control":
        return CONTROL
    return Intervention(name=perturbation, genes=tuple(perturbation.split("_")))


def build(adata: ad.AnnData, n_genes: int, max_cells: int, seed: int) -> CountsDataset:
    """The `n_genes` most variable genes in log-normalized expression, plus every
    measured target gene; at most `max_cells` cells per perturbation. Totals are
    left to the row sums over these features, as for simulated cells."""
    x = sp.csc_matrix(adata.X)
    labels = adata.obs["perturbation"].astype(str).to_numpy()
    totals = np.asarray(x.sum(axis=1)).ravel()
    norm = sp.csc_matrix(x.multiply(1e4 / np.maximum(totals, 1)[:, None]))
    norm.data = np.log1p(norm.data)
    mean = np.asarray(norm.mean(axis=0)).ravel()
    var = np.asarray(norm.multiply(norm).mean(axis=0)).ravel() - mean**2
    names = list(adata.var_names)
    cols = sorted(np.argsort(-var)[:n_genes].tolist())
    targets = {g for p in set(labels) for g in intervention(p).genes}
    where = {g: i for i, g in enumerate(names)}
    cols += sorted(where[g] for g in targets if g in where and where[g] not in cols)

    sub = sp.csr_matrix(x[:, cols])
    rng = np.random.default_rng(seed)
    counts, interventions = {}, {}
    for p in sorted(set(labels)):
        rows = np.flatnonzero(labels == p)
        if len(rows) > max_cells:
            rows = np.sort(rng.choice(rows, max_cells, replace=False))
        counts[p] = torch.as_tensor(sub[rows].toarray(), dtype=torch.float32)
        interventions[p] = intervention(p)
    return CountsDataset(
        name="norman2019",
        features=tuple(names[i] for i in cols),
        counts=counts,
        interventions=interventions,
    )


def combo_split(ds: CountsDataset, frac: float = 0.2, seed: int = 0) -> list[str]:
    """A random `frac` of the double perturbations whose singles are both present."""
    singles = {iv.genes[0] for iv in ds.interventions.values() if len(iv.genes) == 1}
    combos = [
        c
        for c, iv in ds.interventions.items()
        if len(iv.genes) == 2 and set(iv.genes) <= singles
    ]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(combos), generator=g)[: round(frac * len(combos))]
    return sorted(combos[i] for i in idx.tolist())


def make_system(
    ds: CountsDataset,
    *,
    latents: int = 16,
    levels: int = 8,
    width: int = 64,
    layers: int = 2,
    decoder_width: int = 128,
    burn_in: int = 20,
    grad_steps: int | None = 10,
    device: str = "cpu",
) -> CellSystem:
    """A diffusion kernel over unnamed latents, read out by a learned decoder whose
    bias starts at the control cells' log mean counts."""
    schema = latent_schema(latents, levels)
    genes = sorted({g for iv in ds.interventions.values() for g in iv.genes})
    kernel = DiffusionKernel(
        state_schema=schema, genes=tuple(genes), width=width, layers=layers
    )
    observer = DecoderObserver(
        genes=ds.features,
        drivers=tuple(schema.variables),
        width=decoder_width,
        modality=ds.modality,
    )
    observer.set_bias(ds.counts["control"].mean(dim=0).clamp(min=1e-3).log())
    system = CellSystem(
        dynamics=kernel,
        observers={ds.modality: observer},
        burn_in=burn_in,
        grad_steps=grad_steps,
    )
    return system.to(device)
