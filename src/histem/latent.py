from histem.data import CountsDataset
from histem.genes import GeneTable
from histem.models.diffusion import DiffusionKernel
from histem.observers import DecoderObserver
from histem.state import latent_schema
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
    tau: float = 2.0,
    burn_in: int = 20,
    grad_steps: int | None = 10,
    device: str = "cpu",
) -> CellSystem:
    """A diffusion kernel over unnamed latents, read out by a learned decoder whose
    bias starts at the control cells' log mean counts. Both read one gene table
    over the measured and the perturbed genes."""
    schema = latent_schema(latents, levels)
    targets = {g for iv in ds.interventions.values() for g in iv.genes}
    genes = GeneTable(genes=tuple(sorted(targets | set(ds.features))), dim=gene_dim)
    kernel = DiffusionKernel(
        state_schema=schema,
        gene_dim=gene_dim,
        width=width,
        layers=layers,
        tau=tau,
    )
    observer = DecoderObserver(
        genes=ds.features,
        drivers=tuple(schema.variables),
        width=decoder_width,
        gene_dim=gene_dim,
        modality=ds.modality,
    )
    observer.set_bias(ds.counts["control"].mean(dim=0).clamp(min=1e-3).log())
    system = CellSystem(
        dynamics=kernel,
        observers={ds.modality: observer},
        genes=genes,
        burn_in=burn_in,
        grad_steps=grad_steps,
    )
    return system.to(device)
