"""A learned transition kernel: masked discrete diffusion over the next values of
some discrete slots of a cell's state.

Each state variable, each received signal, the context (e.g. the dataset) and each
perturbed gene is a token; a perturbed gene's token is its vector in the system's
`GeneTable` plus a learned vector for the perturbation's kind, so combinations of
perturbations are sets of tokens. A transformer over one cell's tokens passes
messages between them; between cells, messages go through the simulator's signal
routing. A step starts with every variable's next level masked and unmasks them in
`rounds` random chunks, each conditioned on x, the received signals, the perturbed
genes and the levels unmasked so far. Levels are drawn with straight-through
Gumbel-softmax, so states stay on the grid and gradients still flow.
"""

import itertools
from collections.abc import Iterator
from typing import Self

import torch
from pydantic import Field, PrivateAttr, model_validator
from torch import nn

from histem.dynamics import KINDS, Inputs
from histem.genes import GeneTable
from histem.spec import FrozenSpec, Tensor
from histem.state import Population, Slot


class _Net(nn.Module):
    def __init__(
        self,
        n_vars: int,
        n_signals: int,
        contexts: tuple[str, ...],
        gene_dim: int,
        levels: int,
        *,
        width: int,
        layers: int,
        heads: int,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.token = nn.Parameter(0.02 * torch.randn(n_vars + n_signals, width))
        self.context = nn.ParameterDict(
            {c: nn.Parameter(0.02 * torch.randn(width)) for c in contexts}
        )
        self.gene = nn.Linear(gene_dim, width)
        self.kind = nn.Parameter(0.02 * torch.randn(len(KINDS), width))
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

    def extra(self, context: str, genes: torch.Tensor, kind: int) -> torch.Tensor:
        """(1 + k, width) tokens for the context and the perturbed genes, given
        their table vectors (k, gene_dim)."""
        out: torch.Tensor = self.context[context].unsqueeze(0)
        if len(genes):
            out = torch.cat([out, self.gene(genes) + self.kind[kind]])
        return out

    def forward(
        self,
        partial: torch.Tensor,
        x: torch.Tensor,
        received: torch.Tensor,
        extra: torch.Tensor,
    ) -> torch.Tensor:
        """Logits (n, n_vars, levels) given the partial next state
        (n, n_vars, levels + 1) and the `extra` tokens."""
        values = torch.cat([x, received], dim=1).unsqueeze(-1)
        nxt = self.next(partial)
        pad = nxt.new_zeros(x.shape[0], received.shape[1], nxt.shape[-1])
        h = self.token + self.value(values) + torch.cat([nxt, pad], dim=1)
        h = torch.cat([h, extra.expand(x.shape[0], -1, -1)], dim=1)
        out: torch.Tensor = self.logits(self.encoder(h)[:, : self.n_vars])
        return out


class DiffusionKernel(FrozenSpec):
    """A mechanism that reads and writes `slots`."""

    slots: tuple[Slot, ...] = Field(min_length=1)
    contexts: tuple[str, ...] = Field(min_length=1)
    signals: tuple[str, ...] = ()
    gene_dim: int = Field(128, gt=0)  # the bound GeneTable's dim
    width: int = Field(64, gt=0)
    layers: int = Field(2, gt=0)
    heads: int = Field(4, gt=0)
    rounds: int = Field(3, gt=0)
    tau: float = Field(2.0, gt=0)  # Gumbel-softmax temperature
    # empty means freshly initialized; afterwards it holds the live parameters
    weights: dict[str, Tensor] = Field(default_factory=dict)

    _net: _Net = PrivateAttr()
    _invalid: torch.Tensor = PrivateAttr()  # (n_vars, levels) levels a var lacks
    _genes: GeneTable | None = PrivateAttr(None)

    @model_validator(mode="after")
    def _build(self) -> Self:
        if any(s.levels is None for s in self.slots):
            raise ValueError("diffusion slots must be discrete")
        per_var = [s.levels or 0 for s in self.slots for _ in s.variables]
        self._invalid = torch.arange(max(per_var)) >= torch.tensor(per_var)[:, None]
        self._net = _Net(
            len(per_var),
            len(self.signals),
            self.contexts,
            self.gene_dim,
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
    def levels(self) -> int:
        return int(self._invalid.shape[1])

    @property
    def writes(self) -> tuple[Slot, ...]:
        return self.slots

    @property
    def reads(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.slots)

    def _flat(self, pop: Population) -> torch.Tensor:
        return torch.cat([pop.values[s.name] for s in self.slots], dim=1)

    def named_parameters(self) -> Iterator[tuple[str, torch.Tensor]]:
        return iter(self.weights.items())

    def bind_genes(self, table: GeneTable) -> None:
        if table.dim != self.gene_dim:
            raise ValueError(f"gene table dim {table.dim} != gene_dim {self.gene_dim}")
        self._genes = table

    def _logits(
        self,
        partial: torch.Tensor,
        x: torch.Tensor,
        received: torch.Tensor,
        extra: torch.Tensor,
    ) -> torch.Tensor:
        logits: torch.Tensor = self._net(partial, x, received, extra)
        return logits.masked_fill(self._invalid.to(logits.device), float("-inf"))

    def step(self, pop: Population, inputs: Inputs) -> dict[str, torch.Tensor]:
        x = self._flat(pop)
        n, v = x.shape
        iv = inputs.intervention
        if inputs.context not in self.contexts:
            raise KeyError(f"unknown context {inputs.context!r}")
        genes, kind = x.new_zeros(0, self.gene_dim), 0
        if iv.genes:
            if self._genes is None:
                raise RuntimeError("no gene table bound")
            assert iv.kind is not None
            genes, kind = self._genes.embed(iv.genes), KINDS.index(iv.kind)
        extra = self._net.extra(inputs.context, genes, kind)
        partial = x.new_zeros(n, v, self.levels + 1)
        partial[..., -1] = 1.0
        order = torch.rand(n, v, device=x.device).argsort(dim=1).argsort(dim=1)
        bounds = torch.linspace(0, v, self.rounds + 1).round().long().tolist()
        for lo, hi in itertools.pairwise(bounds):
            reveal = ((order >= lo) & (order < hi)).unsqueeze(-1)
            logits = self._logits(partial, x, inputs.signals, extra)
            drawn = nn.functional.gumbel_softmax(logits, tau=self.tau, hard=True)
            drawn = torch.cat([drawn, drawn.new_zeros(n, v, 1)], dim=-1)
            partial = torch.where(reveal, drawn, partial)
        grid = torch.arange(self.levels, device=x.device, dtype=x.dtype)
        nxt = (partial[..., :-1] @ grid).split([s.dim for s in self.slots], dim=1)
        return {s.name: v for s, v in zip(self.slots, nxt, strict=True)}

    def emit_signals(self, pop: Population) -> torch.Tensor:
        if self._net.emit is None:
            return torch.zeros((pop.n, 0), device=pop.device)
        return nn.functional.softplus(self._net.emit(self._flat(pop)))
