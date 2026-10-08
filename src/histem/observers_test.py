import numpy as np
import pytest
from pydantic import ValidationError

from histem.observers import NBCountObserver


def test_weight_shape_must_match_genes_and_drivers() -> None:
    with pytest.raises(ValidationError, match="weights shape"):
        NBCountObserver(
            genes=("g1", "g2"),
            drivers=("a",),
            weights=np.zeros((3, 1)),
            bias=np.zeros(2),
        )


def test_dispersion_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        NBCountObserver(
            genes=("g1",),
            drivers=("a",),
            weights=np.zeros((1, 1)),
            bias=np.zeros(1),
            dispersion=0,
        )
