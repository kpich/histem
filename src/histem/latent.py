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
    ds: CountsDataset,
    *,
    latents: int = 16,
    levels: int = 8,
    width: int = 64,
    layers: int = 2,
    decoder_width: int = 128,
    gene_dim: int = 128,
    rank: int = 16,
    tau: float = 2.0,
    burn_in: int = 20,
    grad_steps: int | None = 10,
    device: str = "cpu",
) -> CellSystem:
    """A diffusion kernel over unnamed latents, plus a continuous per-gene activity
    slot over every table gene that `Perturb` holds at each target's learned
    magnitude and `Regulate` spreads to the other genes. A learned decoder reads
    both; its bias starts at the control cells' log mean counts. All parts read one
    gene table over the measured and the perturbed genes."""
    schema = latent_schema(latents, levels)
    targets = {g for iv in ds.interventions.values() for g in iv.genes}
    genes = GeneTable(genes=tuple(sorted(targets | set(ds.features))), dim=gene_dim)
    kernel = DiffusionKernel(
        slots=schema.slots,
        gene_dim=gene_dim,
        width=width,
        layers=layers,
        tau=tau,
    )
    activity = Slot(name="genes", variables=genes.genes, levels=None)
    perturb = Perturb(slot=activity, gene_dim=gene_dim)
    regulate = Regulate(slot=activity, gene_dim=gene_dim, rank=rank)
    observer = DecoderObserver(
        genes=ds.features,
        drivers=tuple(schema.variables),
        activity=activity.name,
        width=decoder_width,
        gene_dim=gene_dim,
        modality=ds.modality,
    )
    observer.set_bias(ds.counts["control"].mean(dim=0).clamp(min=1e-3).log())
    system = CellSystem(
        dynamics=Composite(
            mechanisms={"regulate": regulate, "perturb": perturb, "latent": kernel}
        ),
        observers={ds.modality: observer},
        genes=genes,
        burn_in=burn_in,
        grad_steps=grad_steps,
    )
    return system.to(device)
