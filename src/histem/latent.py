from histem.data import CountsDataset
from histem.genes import GeneTable
from histem.mechanisms import Composite
from histem.models.diffusion import DiffusionKernel
from histem.models.perturb import Perturb
from histem.models.regulate import Regulate
from histem.observers import DecoderObserver
from histem.state import Slot, latent_schema
from histem.system import CellSystem


def make_system(
    datasets: list[CountsDataset],
    *,
    latents: int = 16,
    levels: int = 8,
    width: int = 64,
    layers: int = 2,
    decoder_width: int = 128,
    gene_dim: int = 128,
    rank: int = 16,
    tau: float = 2.0,
    steps: int = 20,
    grad_steps: int | None = 10,
    device: str = "cpu",
) -> CellSystem:
    """A diffusion kernel over unnamed latents, plus a continuous per-gene activity
    slot over every table gene that `Perturb` holds at each target's learned
    magnitude and `Regulate` spreads to the other genes. A learned decoder per
    modality reads both; each context's bias starts at its control cells' log mean
    counts. All parts read one gene table over every dataset's measured and
    perturbed genes."""
    schema = latent_schema(latents, levels)
    contexts = tuple(dict.fromkeys(ds.context for ds in datasets))
    measured = {
        m: tuple(
            dict.fromkeys(f for ds in datasets if ds.modality == m for f in ds.features)
        )
        for m in dict.fromkeys(ds.modality for ds in datasets)
    }
    targets = {
        g for ds in datasets for iv in ds.interventions.values() for g in iv.genes
    }
    names = targets | {g for fs in measured.values() for g in fs}
    genes = GeneTable(genes=tuple(sorted(names)), dim=gene_dim)
    kernel = DiffusionKernel(
        slots=schema.slots,
        contexts=contexts,
        gene_dim=gene_dim,
        width=width,
        layers=layers,
        tau=tau,
    )
    activity = Slot(name="genes", variables=genes.genes, levels=None)
    perturb = Perturb(slot=activity, gene_dim=gene_dim)
    regulate = Regulate(slot=activity, gene_dim=gene_dim, rank=rank)
    observers = {
        m: DecoderObserver(
            genes=features,
            drivers=tuple(schema.variables),
            contexts=tuple(
                dict.fromkeys(ds.context for ds in datasets if ds.modality == m)
            ),
            activity=activity.name,
            width=decoder_width,
            gene_dim=gene_dim,
            modality=m,
        )
        for m, features in measured.items()
    }
    for ds in datasets:
        if "control" in ds.counts:
            control = ds.counts["control"].mean(dim=0).clamp(min=1e-3).log()
            observers[ds.modality].set_bias(ds.context, ds.features, control)
    system = CellSystem(
        dynamics=Composite(
            mechanisms={"regulate": regulate, "perturb": perturb, "latent": kernel}
        ),
        observers=observers,
        genes=genes,
        steps=steps,
        grad_steps=grad_steps,
    )
    return system.to(device)
