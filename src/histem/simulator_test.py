import pytest
import torch
from pydantic import ValidationError

from histem.models.logic import LogicDynamics
from histem.simulator import Signaling, knn_graph, simulate
from histem.state import Slot, StateSchema


def test_paracrine_signal_reaches_neighbors() -> None:
    schema = StateSchema(slots=(Slot(name="s", variables=("src", "rcv")),))
    dyn = LogicDynamics(
        state_schema=schema,
        rules={"rcv": "in_L >= 0.5"},
        signal_rules={"L": "1.0 if src >= 1 else 0.0"},
        default_rate=1.0,
    )
    pop = schema.empty(3)
    pop.values["s"][0, 0] = 1  # only cell 0 secretes
    positions = torch.tensor([[0.0], [1.0], [10.0]])
    sig = Signaling(autocrine=0.0, paracrine=1.0, neighbors=knn_graph(positions, k=1))
    out, _ = simulate(dyn, pop, 1, signaling=sig)
    assert out.get("rcv").tolist() == [0, 1, 0]  # cell 1's nearest neighbor is cell 0


def test_signaling_rejects_malformed_edges() -> None:
    with pytest.raises(ValidationError, match="shape"):
        Signaling(neighbors=torch.zeros((3, 4), dtype=torch.long))


def test_signaling_rejects_negative_weights() -> None:
    with pytest.raises(ValidationError):
        Signaling(paracrine=-1.0)
