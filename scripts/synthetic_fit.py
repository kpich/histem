"""Fit a diffusion kernel to synthetic snapshots and score it on train conditions
and held-out double KOs, against the truth."""

import argparse
import time

import torch

from histem import synthetic
from histem.learners.gradient import gradient_fit
from histem.suite import Suite
from histem.system import CellSystem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
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
        print(f"{name:>12}: " + "  ".join(f"{k}={v:.3f}" for k, v in fits.items()))

    report("truth", truth)
    system = synthetic.make_kernel_system(device=args.device)
    report("kernel +0", system)
    start = time.perf_counter()
    for done in range(0, args.steps, args.report_every):
        steps = min(args.report_every, args.steps - done)
        gradient_fit(system, fit_suite, steps, lr=args.lr)
        report(f"kernel +{done + steps}", system)
    print(f"fit in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
