import math

from histem import synthetic
from histem.learners.gradient import gradient_fit
from histem.suite import Suite


def test_fit_kernel_on_snapshots() -> None:
    truth = synthetic.make_system().model_copy(update={"burn_in": 3})
    train, test = synthetic.make_dataset(truth, cells_per_condition=10).split(
        ["GATA1_KO"]
    )
    suite = Suite(datasets=[train], cells_per_condition=10, loss_conditions=2)

    system = synthetic.make_kernel_system(width=16, layers=1).model_copy(
        update={"burn_in": 3, "grad_steps": 2}
    )
    gradient_fit(system, suite, 2)

    scores = Suite(datasets=[train, test], cells_per_condition=10).evaluate(system)
    assert all(math.isfinite(v) for v in scores.fit.values())
    assert system.sample(5).shape == (5, len(train.features))
