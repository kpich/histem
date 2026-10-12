from histem.data import CountsDataset
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
    burn_in: int = 20,
    grad_steps: int | None = 10,
    device: str = "cpu",
) -> CellSystem:
    """A diffusion kernel over unnamed latents, read out by a learned decoder whose
    bias starts at the control cells' log mean counts. The kernel knows every gene
    the dataset perturbs."""
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
