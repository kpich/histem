import pytest
from pydantic import ValidationError

from histem.dynamics import Intervention


def test_kind_is_set_iff_there_are_genes() -> None:
    with pytest.raises(ValidationError, match="kind"):
        Intervention(name="A", genes=("A",))
    with pytest.raises(ValidationError, match="kind"):
        Intervention(kind="activate")
