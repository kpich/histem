import pytest
import torch
from pydantic import ValidationError

from histem.dynamics import Inputs
from histem.simulator import Signaling, knn_graph, simulate
from histem.spec import FrozenSpec
from histem.state import Population, Slot, StateSchema

SCHEMA = StateSchema(slots=(Slot(name="s", variables=("src", "rcv")),))


class Relay(FrozenSpec):
    """`src` cells secrete L; `rcv` turns on where L arrives."""

    @property
    def state_schema(self) -> StateSchema:
        return SCHEMA

    def step(self, pop: Population, inputs: Inputs) -> Population:
        out = pop.copy()
        out.values["s"][:, 1] = (inputs.signals[:, 0] >= 0.5).to(torch.int8)
        return out

    def emit_signals(self, pop: Population) -> torch.Tensor:
        return pop.get("src").float().unsqueeze(1)


def test_paracrine_signal_reaches_neighbors() -> None:
    pop = SCHEMA.empty(3)
    pop.values["s"][0, 0] = 1  # only cell 0 secretes
    positions = torch.tensor([[0.0], [1.0], [10.0]])
    sig = Signaling(autocrine=0.0, paracrine=1.0, neighbors=knn_graph(positions, k=1))
    out = simulate(Relay(), pop, 1, signaling=sig)
    assert out.get("rcv").tolist() == [0, 1, 0]  # cell 1's nearest neighbor is cell 0


def test_signaling_rejects_malformed_edges() -> None:
    with pytest.raises(ValidationError, match="shape"):
        Signaling(neighbors=torch.zeros((3, 4), dtype=torch.long))


def test_signaling_rejects_negative_weights() -> None:
    with pytest.raises(ValidationError):
        Signaling(paracrine=-1.0)
