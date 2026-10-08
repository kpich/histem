"""Can search recover a known program from its simulated Perturb-seq data?

Starts from the null program (every variable holds its value) and hill-climbs with
random rule edits. Baseline for any smarter Proposer (LLM edits, distillation, ...).

    uv run scripts/synthetic_recovery.py --iters 500
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from histem import synthetic
from histem.learners.search import RandomLogicEdit, hill_climb
from histem.models.logic import LogicDynamics
from histem.suite import Suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--complexity-weight", type=float, default=1e-3)
    ap.add_argument("--held-out", nargs="*", default=["FLI1_KO+KLF1_KO", "PU1_KO+CEBPA_KO"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    truth = synthetic.make_world()
    train, test = synthetic.make_dataset(truth).split(args.held_out)
    suite = Suite([train], complexity_weight=args.complexity_weight)
    test_suite = Suite([test], complexity_weight=args.complexity_weight)

    null = LogicDynamics(
        synthetic.SCHEMA,
        {v: v for v in synthetic.SCHEMA.variables},
        {s: "0.0" for s in truth.dynamics.signal_rules},
        truth.dynamics.rates,
    )
    start = replace(truth, dynamics=null)

    for name, w in [("truth", truth), ("null", start)]:
        s = suite.evaluate(w)
        print(f"{name:>6}: objective={s.objective:.4f} dl={s.description_length:.0f} "
              f"held-out fit={test_suite.evaluate(w).by_dataset()}")

    found, scores, log = hill_climb(
        start, suite, RandomLogicEdit(args.seed), args.iters, args.seed, verbose=True
    )
    print(f"\nfound: objective={scores.objective:.4f}, accepted {len(log.accepted) - 1} edits")
    print(f"held-out fit: {test_suite.evaluate(found).by_dataset()}")
    print("\n--- induced program ---\n" + found.dynamics.to_text())
    print("\n--- true program ---\n" + truth.dynamics.to_text())


if __name__ == "__main__":
    main()
