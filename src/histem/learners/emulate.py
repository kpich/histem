"""Train a learned kernel to emulate a program from its recorded transitions."""

from collections.abc import Iterator
from typing import Protocol

import torch

from histem.dynamics import Inputs, Intervention, Transitions
from histem.learners.gradient import FitLog
from histem.state import StateSchema
from histem.system import CellSystem


class Emulator(Protocol):
    def parameters(self) -> Iterator[torch.Tensor]: ...

    def transition_loss(self, batch: Transitions) -> torch.Tensor: ...


def random_interventions(
    schema: StateSchema, n: int, max_clamps: int = 2
) -> list[Intervention]:
    """n interventions, each clamping 0..max_clamps random variables to random
    levels."""
    variables = schema.variables
    out = []
    for i in range(n):
        k = int(torch.randint(0, max_clamps + 1, ()))
        picked = torch.randperm(len(variables))[:k].tolist()
        clamps = []
        for j in picked:
            levels = schema.slot(schema.locate(variables[j])[0]).levels
            clamps.append((variables[j], float(torch.randint(0, levels, ()))))
        out.append(Intervention(name=f"random{i}", clamps=tuple(clamps)))
    return out


@torch.no_grad()
def record(
    system: CellSystem, interventions: list[Intervention], cells: int, steps: int
) -> Transitions:
    """Every step of `cells` cells over `steps` steps from the system's init,
    under each intervention."""
    parts = []
    for iv in interventions:
        dyn = system.dynamics.intervene(iv)
        pop = system.init.sample(dyn.state_schema, cells, system.device)
        iv.apply(pop)
        for _ in range(steps):
            emitted = dyn.emit_signals(pop)
            received = system.signaling.route(emitted)
            nxt = dyn.step(pop, Inputs(received, iv))
            parts.append(Transitions(pop.flat(), received, emitted, nxt.flat()))
            pop = nxt
            iv.apply(pop)
    return Transitions(
        *(
            torch.cat([getattr(p, f) for p in parts])
            for f in ("x", "received", "emitted", "next")
        )
    )


def fit_emulator(
    emulator: Emulator,
    data: Transitions,
    steps: int,
    *,
    batch_size: int = 1024,
    lr: float = 1e-3,
    verbose: bool = False,
) -> FitLog:
    """Adam on minibatches of `emulator.transition_loss`. Updates in place."""
    opt = torch.optim.Adam(list(emulator.parameters()), lr=lr)
    log = FitLog()
    for it in range(1, steps + 1):
        idx = torch.randint(0, len(data), (batch_size,), device=data.x.device)
        opt.zero_grad()
        loss = emulator.transition_loss(data.batch(idx))
        loss.backward()
        opt.step()
        log.losses.append(loss.item())
        if verbose and (it % 100 == 0 or it == steps):
            print(f"[{it}] transition loss={log.losses[-1]:.4f}")
    return log
