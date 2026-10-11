"""Emulate programs with a diffusion kernel, then fine-tune on snapshots.

Scores, on train conditions and held-out double KOs:
- the truth and a wrong draft of it, run as programs
- a kernel pretrained on the truth's transitions (emulation fidelity)
- a kernel pretrained on the draft's transitions, before and after fine-tuning
  on the truth's snapshots
- a kernel fine-tuned on the snapshots from scratch
"""

import argparse
import time

import torch

from histem import synthetic
from histem.learners.emulate import fit_emulator, random_interventions, record
from histem.learners.gradient import gradient_fit
from histem.models.diffusion import DiffusionKernel
from histem.suite import Suite
from histem.system import CellSystem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-steps", type=int, default=4000)
    ap.add_argument("--finetune-steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--report-every", type=int, default=100)
    ap.add_argument("--cells", type=int, default=128)
    ap.add_argument("--conditions", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--held-out", nargs="*", default=["FLI1_KO+KLF1_KO", "PU1_KO+CEBPA_KO"]
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    truth = synthetic.make_system()
    train, test = synthetic.make_dataset(truth).split(args.held_out)
    fit_suite = Suite(
        datasets=[train],
        cells_per_condition=args.cells,
        loss_conditions=args.conditions,
    )
    suite = Suite(datasets=[train, test])

    def report(name: str, system: CellSystem) -> None:
        fits = suite.evaluate(system).by_dataset()
        print(f"{name:>16}: " + "  ".join(f"{k}={v:.3f}" for k, v in fits.items()))

    def emulator(program: synthetic.ForkProgram) -> CellSystem:
        source = synthetic.make_system(program)
        data = record(source, random_interventions(synthetic.SCHEMA, 64), 512, 60)
        system = synthetic.make_emulator_system(device=args.device)
        start = time.perf_counter()
        assert isinstance(system.dynamics, DiffusionKernel)
        fit_emulator(system.dynamics, data, args.pretrain_steps)
        print(f"pretrained in {time.perf_counter() - start:.0f}s")
        return system

    def finetune(name: str, system: CellSystem) -> None:
        start = time.perf_counter()
        for done in range(0, args.finetune_steps, args.report_every):
            steps = min(args.report_every, args.finetune_steps - done)
            gradient_fit(system, fit_suite, steps, lr=args.lr)
            report(f"{name} +{done + steps}", system)
        print(f"fine-tuned in {time.perf_counter() - start:.0f}s")

    report("truth", truth)
    report("draft program", synthetic.make_system(synthetic.DRAFT))
    report("emulated truth", emulator(synthetic.ForkProgram()))

    draft = emulator(synthetic.DRAFT)
    report("emulated draft", draft)
    finetune("draft", draft)

    scratch = synthetic.make_emulator_system(device=args.device)
    report("scratch", scratch)
    finetune("scratch", scratch)


if __name__ == "__main__":
    main()
