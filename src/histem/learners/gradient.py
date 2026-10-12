import torch
from pydantic import Field

from histem.spec import Spec
from histem.suite import Suite
from histem.system import CellSystem


class FitLog(Spec):
    losses: list[float] = Field(default_factory=list)


def gradient_fit(
    system: CellSystem,
    suite: Suite,
    steps: int,
    *,
    lr: float = 1e-2,
    clip: float | None = 1.0,
    verbose: bool = False,
) -> FitLog:
    """Adam on `suite.loss`, with the gradient norm clipped to `clip`. Updates the
    system's parameters in place."""
    params = list(system.parameters())
    if not params:
        raise ValueError("system has no learnable parameters")
    opt = torch.optim.Adam(params, lr=lr)
    log = FitLog()
    for it in range(1, steps + 1):
        opt.zero_grad()
        loss = suite.loss(system)
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        log.losses.append(loss.item())
        if verbose and (it % 10 == 0 or it == steps):
            print(f"[{it}] loss={log.losses[-1]:.4f}")
    return log
