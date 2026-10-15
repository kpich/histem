"""A learned transition kernel: masked discrete diffusion over a cell's next state.

Each state variable, each received signal and each perturbed gene is a token; a
perturbed gene's token is its vector in the system's `GeneTable` plus a learned
vector for the perturbation's kind, so combinations of perturbations are sets of
tokens. A transformer over one cell's tokens passes messages between them;
between cells, messages go through the simulator's signal routing. A step starts
with every variable's next level masked and unmasks them in `rounds` random
chunks, each conditioned on x, the received signals, the perturbed genes and the
levels unmasked so far. Levels are drawn with straight-through Gumbel-softmax, so
states stay on the grid and gradients still flow.
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
from histem.state import Population, StateSchema


class _Net(nn.Module):
    def __init__(
        self,
        n_vars: int,
        n_signals: int,
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

    def forward(
        self,
        partial: torch.Tensor,
        x: torch.Tensor,
        received: torch.Tensor,
        genes: torch.Tensor,
        kind: int,
    ) -> torch.Tensor:
        """Logits (n, n_vars, levels) given the partial next state
        (n, n_vars, levels + 1) and the perturbed genes' table vectors (k, gene_dim)."""
        values = torch.cat([x, received], dim=1).unsqueeze(-1)
        nxt = self.next(partial)
        pad = nxt.new_zeros(x.shape[0], received.shape[1], nxt.shape[-1])
        h = self.token + self.value(values) + torch.cat([nxt, pad], dim=1)
        if len(genes):
            g = self.gene(genes) + self.kind[kind]
            h = torch.cat([h, g.expand(x.shape[0], -1, -1)], dim=1)
        out: torch.Tensor = self.logits(self.encoder(h)[:, : self.n_vars])
        return out


class DiffusionKernel(FrozenSpec):
    state_schema: StateSchema
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
        per_var = [s.levels for s in self.state_schema.slots for _ in s.variables]
        self._invalid = torch.arange(max(per_var)) >= torch.tensor(per_var)[:, None]
        self._net = _Net(
            len(per_var),
            len(self.signals),
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
        genes: torch.Tensor,
        kind: int,
    ) -> torch.Tensor:
        logits: torch.Tensor = self._net(partial, x, received, genes, kind)
        return logits.masked_fill(self._invalid.to(logits.device), float("-inf"))

    def step(self, pop: Population, inputs: Inputs) -> Population:
        x = pop.flat()
        n, v = x.shape
        iv = inputs.intervention
        genes, kind = x.new_zeros(0, self.gene_dim), 0
        if iv.genes:
            if self._genes is None:
                raise RuntimeError("no gene table bound")
            assert iv.kind is not None
            genes, kind = self._genes.embed(iv.genes), KINDS.index(iv.kind)
        partial = x.new_zeros(n, v, self.levels + 1)
        partial[..., -1] = 1.0
        order = torch.rand(n, v, device=x.device).argsort(dim=1).argsort(dim=1)
        bounds = torch.linspace(0, v, self.rounds + 1).round().long().tolist()
        for lo, hi in itertools.pairwise(bounds):
            reveal = ((order >= lo) & (order < hi)).unsqueeze(-1)
            logits = self._logits(partial, x, inputs.signals, genes, kind)
            drawn = nn.functional.gumbel_softmax(logits, tau=self.tau, hard=True)
            drawn = torch.cat([drawn, drawn.new_zeros(n, v, 1)], dim=-1)
            partial = torch.where(reveal, drawn, partial)
        grid = torch.arange(self.levels, device=x.device, dtype=x.dtype)
        return pop.with_flat(partial[..., :-1] @ grid)

    def emit_signals(self, pop: Population) -> torch.Tensor:
        if self._net.emit is None:
            return torch.zeros((pop.n, 0), device=pop.device)
        return nn.functional.softplus(self._net.emit(pop.flat()))
