"""Gradient-fit neural dynamics on synthetic data; compare to the true program."""

import argparse
import time

import torch

from histem import synthetic
from histem.learners.gradient import gradient_fit
from histem.suite import Suite


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--cells", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--held-out", nargs="*", default=["FLI1_KO+KLF1_KO", "PU1_KO+CEBPA_KO"]
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    truth = synthetic.make_system()
    train, test = synthetic.make_dataset(truth).split(args.held_out)
    fit_suite = Suite(datasets=[train], cells_per_condition=args.cells)
    suite = Suite(datasets=[train, test])
    neural = synthetic.make_neural_system(device=args.device)

    for name, s in [("truth", truth), ("init", neural)]:
        print(f"{name:>6}: fit={suite.evaluate(s).by_dataset()}")
    start = time.perf_counter()
    gradient_fit(neural, fit_suite, args.steps, lr=args.lr, verbose=True)
    secs = time.perf_counter() - start
    print(f"\n{args.steps} steps in {secs:.0f}s on {args.device}")
    print(f"fitted: fit={suite.evaluate(neural).by_dataset()}")


if __name__ == "__main__":
    main()
