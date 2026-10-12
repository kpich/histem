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
    callback: Callable[[int, float, dict[str, float]], None] | None = None,
) -> list[float]:
    """Adam on `suite.loss`, with the gradient norm clipped to `clip`. Updates the
    system's parameters in place and returns the losses. `callback(step, loss,
    grad_norms)` runs after each step; `grad_norms` are before clipping, per named
    parameter plus "total"."""
    named = dict(system.named_parameters())
    if not named:
        raise ValueError("system has no learnable parameters")
    params = list(named.values())
    opt = torch.optim.Adam(params, lr=lr)
    losses = []
    for it in range(1, steps + 1):
        opt.zero_grad()
        loss = suite.loss(system)
        loss.backward()
        norms = torch.stack(
            [p.grad.norm() if p.grad is not None else p.new_zeros(()) for p in params]
        )
        total = torch.nn.utils.clip_grad_norm_(
            params, float("inf") if clip is None else clip
        )
        opt.step()
        losses.append(loss.item())
        if callback is not None:
            values = torch.cat([norms, total.reshape(1)]).tolist()
            callback(it, losses[-1], dict(zip([*named, "total"], values, strict=True)))
    return losses
