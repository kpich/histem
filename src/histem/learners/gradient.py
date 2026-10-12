from collections.abc import Callable

import torch

from histem.suite import Suite
from histem.system import CellSystem


def gradient_fit(
    system: CellSystem,
    suite: Suite,
    steps: int,
    *,
    lr: float = 1e-2,
    clip: float | None = 1.0,
    callback: Callable[[int, float, float], None] | None = None,
) -> list[float]:
    """Adam on `suite.loss`, with the gradient norm clipped to `clip`. Updates the
    system's parameters in place and returns the losses. `callback(step, loss,
    grad_norm)` runs after each step."""
    params = list(system.parameters())
    if not params:
        raise ValueError("system has no learnable parameters")
    opt = torch.optim.Adam(params, lr=lr)
    losses = []
    for it in range(1, steps + 1):
        opt.zero_grad()
        loss = suite.loss(system)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            params, float("inf") if clip is None else clip
        )
        opt.step()
        losses.append(loss.item())
        if callback is not None:
            callback(it, losses[-1], float(norm))
    return losses
