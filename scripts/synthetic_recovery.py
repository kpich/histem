"""Hill-climb from the null program on synthetic data; compare to the true program."""

import argparse

from histem import synthetic
from histem.learners.search import RandomLogicEdit, hill_climb
from histem.models.logic import LogicDynamics
from histem.suite import Suite


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--complexity-weight", type=float, default=1e-3)
    ap.add_argument(
        "--held-out", nargs="*", default=["FLI1_KO+KLF1_KO", "PU1_KO+CEBPA_KO"]
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    truth = synthetic.make_system()
    assert isinstance(truth.dynamics, LogicDynamics)
    train, test = synthetic.make_dataset(truth).split(args.held_out)
    suite = Suite(datasets=[train], complexity_weight=args.complexity_weight)
    test_suite = Suite(datasets=[test], complexity_weight=args.complexity_weight)

    null = LogicDynamics(
        state_schema=synthetic.SCHEMA,
        rules={v: v for v in synthetic.SCHEMA.variables},
        signal_rules=dict.fromkeys(truth.dynamics.signal_rules, "0.0"),
        rates=truth.dynamics.rates,
    )
    start = truth.with_dynamics(null)

    for name, w in [("truth", truth), ("null", start)]:
        s = suite.evaluate(w)
        print(
            f"{name:>6}: objective={s.objective:.4f} dl={s.description_length:.0f} "
            f"held-out fit={test_suite.evaluate(w).by_dataset()}"
        )

    found, scores, log = hill_climb(
        start,
        suite,
        RandomLogicEdit(seed=args.seed),
        args.iters,
        seed=args.seed,
        verbose=True,
    )
    n_edits = len(log.accepted) - 1
    print(f"\nfound: objective={scores.objective:.4f}, accepted {n_edits} edits")
    print(f"held-out fit: {test_suite.evaluate(found).by_dataset()}")
    assert isinstance(found.dynamics, LogicDynamics)
    print("\n--- induced program ---\n" + found.dynamics.to_text())
    print("\n--- true program ---\n" + truth.dynamics.to_text())


if __name__ == "__main__":
    main()
