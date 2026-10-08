import numpy as np

from histem.models.logic import LogicDynamics
from histem.simulator import Signaling, knn_graph, simulate
from histem.state import Slot, StateSchema


def test_paracrine_signal_reaches_neighbors() -> None:
    schema = StateSchema((Slot("s", ("src", "rcv"), levels=2),))
    dyn = LogicDynamics(
        schema,
        {"rcv": "in_L >= 0.5"},
        {"L": "1.0 if src >= 1 else 0.0"},
        default_rate=1.0,
    )
    pop = schema.empty(3)
    pop.values["s"][0, 0] = 1  # only cell 0 secretes
    positions = np.array([[0.0], [1.0], [10.0]])
    sig = Signaling(autocrine=0.0, paracrine=1.0, neighbors=knn_graph(positions, k=1))
    out, _ = simulate(dyn, pop, 1, np.random.default_rng(0), signaling=sig)
    assert out.get("rcv").tolist() == [0, 1, 0]  # cell 1's nearest neighbor is cell 0
