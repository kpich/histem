"""A learned transition kernel: masked discrete diffusion over a cell's next state.

Each state variable and each received signal is a token. A transformer over one
cell's tokens passes messages between them; between cells, messages go through
the simulator's signal routing. A step starts with every variable's next level
masked and unmasks them in `rounds` random chunks, each conditioned on x, the
received signals and the levels unmasked so far. Levels are drawn with
straight-through Gumbel-softmax, so states stay on the grid and gradients still
flow.
"""

import itertools
from collections.abc import Iterator
from typing import Self

import torch
from pydantic import Field, PrivateAttr, model_validator
from torch import nn

from histem.dynamics import Inputs, Intervention
from histem.spec import FrozenSpec, Tensor
from histem.state import Population, StateSchema


class _Net(nn.Module):
    def __init__(
        self,
        n_vars: int,
        n_signals: int,
        levels: int,
        *,
        width: int,
        layers: int,
        heads: int,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.token = nn.Parameter(0.02 * torch.randn(n_vars + n_signals, width))
        self.value = nn.Linear(1, width)
        self.next = nn.Linear(levels + 1, width)  # one-hot level, last column = mask
        layer = nn.TransformerEncoderLayer(
            width, heads, 2 * width, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.logits = nn.Linear(width, levels)
        self.emit = (
            nn.Sequential(
                nn.Linear(n_vars, width), nn.SiLU(), nn.Linear(width, n_signals)
            )
            if n_signals
            else None
        )

    def forward(
        self, partial: torch.Tensor, x: torch.Tensor, received: torch.Tensor
    ) -> torch.Tensor:
        """Logits (n, n_vars, levels) given the partial next state
        (n, n_vars, levels + 1)."""
        values = torch.cat([x, received], dim=1).unsqueeze(-1)
        nxt = self.next(partial)
        pad = nxt.new_zeros(x.shape[0], received.shape[1], nxt.shape[-1])
        h = self.token + self.value(values) + torch.cat([nxt, pad], dim=1)
        out: torch.Tensor = self.logits(self.encoder(h)[:, : self.n_vars])
        return out


class DiffusionKernel(FrozenSpec):
    state_schema: StateSchema
    signals: tuple[str, ...] = ()
    width: int = Field(64, gt=0)
    layers: int = Field(2, gt=0)
    heads: int = Field(4, gt=0)
    rounds: int = Field(3, gt=0)
    # empty means freshly initialized; afterwards it holds the live parameters
    weights: dict[str, Tensor] = Field(default_factory=dict)

    _net: _Net = PrivateAttr()
    _invalid: torch.Tensor = PrivateAttr()  # (n_vars, levels) levels a var lacks

    @model_validator(mode="after")
    def _build(self) -> Self:
        if bad := [s.name for s in self.state_schema.slots if s.kind != "continuous"]:
            raise ValueError(f"DiffusionKernel needs continuous slots, got {bad}")
        per_var = [s.levels for s in self.state_schema.slots for _ in s.variables]
        self._invalid = torch.arange(max(per_var)) >= torch.tensor(per_var)[:, None]
        self._net = _Net(
            len(per_var),
            len(self.signals),
            max(per_var),
            width=self.width,
            layers=self.layers,
            heads=self.heads,
        )
        if self.weights:
            self._net.load_state_dict(self.weights, assign=True)
        self.weights.clear()
        self.weights.update(self._net.named_parameters())
        return self

    @property
    def signal_names(self) -> tuple[str, ...]:
        return self.signals

    @property
    def levels(self) -> int:
        return int(self._invalid.shape[1])

    def parameters(self) -> Iterator[torch.Tensor]:
        return iter(self.weights.values())

    def _logits(
        self, partial: torch.Tensor, x: torch.Tensor, received: torch.Tensor
    ) -> torch.Tensor:
        logits: torch.Tensor = self._net(partial, x, received)
        return logits.masked_fill(self._invalid.to(logits.device), float("-inf"))

    def step(self, pop: Population, inputs: Inputs) -> Population:
        x = pop.flat()
        n, v = x.shape
        partial = x.new_zeros(n, v, self.levels + 1)
        partial[..., -1] = 1.0
        order = torch.rand(n, v, device=x.device).argsort(dim=1).argsort(dim=1)
        bounds = torch.linspace(0, v, self.rounds + 1).round().long().tolist()
        for lo, hi in itertools.pairwise(bounds):
            reveal = ((order >= lo) & (order < hi)).unsqueeze(-1)
            logits = self._logits(partial, x, inputs.signals)
            drawn = nn.functional.gumbel_softmax(logits, hard=True)
            drawn = torch.cat([drawn, drawn.new_zeros(n, v, 1)], dim=-1)
            partial = torch.where(reveal, drawn, partial)
        grid = torch.arange(self.levels, device=x.device, dtype=x.dtype)
        return pop.with_flat(partial[..., :-1] @ grid)

    def emit_signals(self, pop: Population) -> torch.Tensor:
        if self._net.emit is None:
            return torch.zeros((pop.n, 0), device=pop.device)
        return nn.functional.softplus(self._net.emit(pop.flat()))

    def intervene(self, intervention: Intervention) -> Self:
        return self

    def description_length(self) -> float:
        """32 bits per parameter."""
        return float(sum(p.numel() for p in self.weights.values())) * 32.0
