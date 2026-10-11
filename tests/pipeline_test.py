import math

from histem import synthetic
from histem.dynamics import CONTROL
from histem.learners.emulate import fit_emulator, record
from histem.learners.gradient import gradient_fit
from histem.models.diffusion import DiffusionKernel
from histem.suite import Suite


def test_emulate_program_then_finetune_on_snapshots() -> None:
    truth = synthetic.make_system().model_copy(update={"burn_in": 3})
    train, test = synthetic.make_dataset(truth, cells_per_condition=10).split(
        ["GATA1_KO"]
    )
    suite = Suite(datasets=[train], cells_per_condition=10, loss_conditions=2)

    system = synthetic.make_emulator_system(width=16, layers=1).model_copy(
        update={"burn_in": 3, "grad_steps": 2}
    )
    assert isinstance(system.dynamics, DiffusionKernel)
    data = record(synthetic.make_system(synthetic.DRAFT), [CONTROL], 16, 3)
    fit_emulator(system.dynamics, data, 2, batch_size=8)
    gradient_fit(system, suite, 2)

    scores = Suite(datasets=[train, test], cells_per_condition=10).evaluate(system)
    assert all(math.isfinite(v) for v in scores.fit.values())
    assert system.sample(5).shape == (5, len(train.features))
