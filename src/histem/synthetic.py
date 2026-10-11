"""Toy ground truth: GATA1/PU1 fork, chromatin gates, MYC/mito loop, IL signal."""

import torch
from pydantic import Field

from histem.data import CountsDataset
from histem.dynamics import CONTROL, Inputs, Intervention
from histem.models.diffusion import DiffusionKernel
from histem.observers import NBCountObserver
from histem.rng import seeded
from histem.simulator import Signaling
from histem.spec import FrozenSpec
from histem.state import Population, Slot, StateSchema
from histem.system import CellSystem

TFS = ("GATA1", "PU1", "FLI1", "KLF1", "CEBPA", "MYC")

SCHEMA = StateSchema(
    slots=(
        Slot(name="expr", variables=TFS, levels=3, observed=True),
        Slot(name="chromatin", variables=("acc_GATA1", "acc_CEBPA"), levels=2),
        Slot(name="mito", variables=("mito_cn",), levels=3),
    )
)


class ForkProgram(FrozenSpec):
    """Asynchronous multi-valued logic. Each step, each variable fires with
    probability `rates` and moves one level toward its target.

    The flags switch off mechanisms, to make wrong drafts of the truth.
    """

    pu1_represses_gata1: bool = True
    il_opens_cebpa: bool = True
    rates: dict[str, float] = Field(
        default_factory=lambda: {"acc_GATA1": 0.05, "acc_CEBPA": 0.05, "mito_cn": 0.1}
    )
    default_rate: float = 0.5

    @property
    def state_schema(self) -> StateSchema:
        return SCHEMA

    @property
    def signal_names(self) -> tuple[str, ...]:
        return ("IL",)

    def targets(
        self, v: dict[str, torch.Tensor], il: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        gata1 = (v["acc_GATA1"] >= 1) & ((v["GATA1"] >= 1) | (v["FLI1"] >= 1))
        if self.pu1_represses_gata1:
            gata1 &= v["PU1"] < 2
        acc_cebpa = v["acc_CEBPA"] >= 1
        if self.il_opens_cebpa:
            acc_cebpa |= il >= 0.5
        on = {
            "GATA1": gata1,
            "PU1": ((v["PU1"] >= 1) | (v["CEBPA"] >= 1)) & (v["GATA1"] < 2),
            "FLI1": (v["GATA1"] >= 1) & (v["KLF1"] < 1),
            "KLF1": (v["GATA1"] >= 2) & (v["FLI1"] < 2),
            "CEBPA": (v["acc_CEBPA"] >= 1) & (v["PU1"] >= 1),
        }
        out = {k: 2.0 * c for k, c in on.items()}
        myc_high = (v["mito_cn"] >= 1) & (v["KLF1"] < 2) & (v["CEBPA"] < 2)
        out["MYC"] = 1.0 + myc_high.float()
        out["mito_cn"] = 1.0 + (v["MYC"] >= 2).float()
        out["acc_GATA1"] = (v["CEBPA"] < 2).float()
        out["acc_CEBPA"] = acc_cebpa.float()
        return out

    def step(self, pop: Population, inputs: Inputs) -> Population:
        x = pop.flat()
        v = dict(zip(SCHEMA.variables, x.unbind(1), strict=True))
        targets = self.targets(v, inputs.signals[:, 0])
        target = torch.stack([targets[name] for name in SCHEMA.variables], 1)
        rate = torch.tensor(
            [self.rates.get(name, self.default_rate) for name in SCHEMA.variables],
            device=x.device,
        )
        fire = torch.rand_like(x) < rate
        return pop.with_flat(torch.where(fire, x + torch.sign(target - x), x))

    def emit_signals(self, pop: Population) -> torch.Tensor:
        return (pop.get("PU1") >= 2).float().unsqueeze(1)

    def intervene(self, intervention: Intervention) -> "ForkProgram":
        return self

    def description_length(self) -> float:
        return 0.0


# the truth without PU1 -| GATA1 and without IL opening CEBPA chromatin
DRAFT = ForkProgram(pu1_represses_gata1=False, il_opens_cebpa=False)

DRIVERS = (*TFS, "mito_cn")


def make_observer(
    genes_per_driver: int = 4, housekeeping: int = 20, seed: int = 0
) -> NBCountObserver:
    g = torch.Generator().manual_seed(seed)

    def uniform(lo: float, hi: float, n: int = 1) -> torch.Tensor:
        return lo + (hi - lo) * torch.rand(n, generator=g)

    genes, rows = [], []
    for j, d in enumerate(DRIVERS):
        for k in range(genes_per_driver):
            genes.append(f"{d}_t{k}")
            w = torch.zeros(len(DRIVERS))
            w[j] = uniform(1.5, 3.0)
            if torch.rand(1, generator=g) < 0.3:
                w[torch.randint(len(DRIVERS), (1,), generator=g)] += uniform(-1.0, 1.0)
            rows.append(w)
    for k in range(housekeeping):
        genes.append(f"HK{k}")
        rows.append(torch.zeros(len(DRIVERS)))
    return NBCountObserver(
        genes=tuple(genes),
        drivers=DRIVERS,
        weights=torch.stack(rows),
        bias=uniform(-1.0, 1.5, len(genes)),
    )


def make_system(program: ForkProgram | None = None) -> CellSystem:
    return CellSystem(
        dynamics=program or ForkProgram(),
        observers={"rna": make_observer()},
        burn_in=60,
        signaling=Signaling(autocrine=0.5, endocrine=0.5),
    )


def make_emulator_system(
    width: int = 64,
    layers: int = 2,
    rounds: int = 3,
    grad_steps: int | None = 10,
    device: str = "cpu",
) -> CellSystem:
    """A DiffusionKernel on the relaxed schema, under the true observer."""
    kernel = DiffusionKernel(
        state_schema=SCHEMA.relaxed(),
        signals=("IL",),
        width=width,
        layers=layers,
        rounds=rounds,
    )
    system = make_system().model_copy(
        update={"dynamics": kernel, "grad_steps": grad_steps}
    )
    return system.to(device)


def perturbations() -> dict[str, Intervention]:
    out = {"control": CONTROL}
    for tf in TFS:
        out[f"{tf}_KO"] = Intervention(name=f"{tf}_KO", clamps=((tf, 0),))
        out[f"{tf}_OE"] = Intervention(name=f"{tf}_OE", clamps=((tf, 2),))
    for a, b in [("GATA1", "PU1"), ("FLI1", "KLF1"), ("PU1", "CEBPA")]:
        name = f"{a}_KO+{b}_KO"
        out[name] = Intervention(name=name, clamps=((a, 0), (b, 0)))
    return out


def make_dataset(
    system: CellSystem | None = None, cells_per_condition: int = 300, seed: int = 1
) -> CountsDataset:
    system = system or make_system()
    interventions = perturbations()
    with torch.no_grad(), seeded(seed):
        counts = {
            name: system.sample(cells_per_condition, iv)
            for name, iv in interventions.items()
        }
    return CountsDataset(
        name="synthetic_fork",
        features=system.observers["rna"].features,
        counts=counts,
        interventions=interventions,
    )
