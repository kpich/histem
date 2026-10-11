import torch

from histem import synthetic
from histem.dynamics import Intervention
from histem.learners.emulate import fit_emulator, random_interventions, record
from histem.models.diffusion import DiffusionKernel


def test_random_interventions_clamp_valid_levels() -> None:
    torch.manual_seed(0)
    for iv in random_interventions(synthetic.SCHEMA, 50, max_clamps=2):
        assert len(iv.clamps) <= 2
        for variable, value in iv.clamps:
            slot = synthetic.SCHEMA.slot(synthetic.SCHEMA.locate(variable)[0])
            assert 0 <= value < slot.levels


def test_record_keeps_clamps_on_inputs_but_not_outputs() -> None:
    torch.manual_seed(0)
    ko = Intervention(name="ko", clamps=(("PU1", 2.0),))
    data = record(synthetic.make_system(), [ko, Intervention()], 30, 4)
    assert len(data) == 2 * 30 * 4
    assert data.x.shape == data.next.shape == (240, len(synthetic.SCHEMA.variables))
    assert data.received.shape == data.emitted.shape == (240, 1)
    col = synthetic.SCHEMA.variables.index("PU1")
    assert (data.x[:120, col] == 2).all()


def test_fit_emulator_updates_parameters() -> None:
    torch.manual_seed(0)
    kernel = synthetic.make_emulator_system(width=16, layers=1).dynamics
    assert isinstance(kernel, DiffusionKernel)
    data = record(synthetic.make_system(), [Intervention()], 16, 3)
    before = [p.detach().clone() for p in kernel.parameters()]
    log = fit_emulator(kernel, data, 2, batch_size=8)
    assert len(log.losses) == 2
    after = list(kernel.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))
