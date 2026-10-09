from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from histem.spec import Spec, Tagged, Tensor


class Holder(Spec):
    x: Tensor
    ys: dict[str, Tensor]


class Outer(Spec):
    inner: Tagged[object]


def test_tensors_go_to_the_tensor_file(tmp_path: Path) -> None:
    h = Holder(x=torch.arange(6).reshape(2, 3), ys={"a": torch.ones(2)})
    h.save(tmp_path)
    assert "tensor" in (tmp_path / "config.json").read_text()
    loaded = Holder.load(tmp_path)
    assert torch.equal(loaded.x, h.x)
    assert torch.equal(loaded.ys["a"], h.ys["a"])


def test_tagged_field_restores_its_class(tmp_path: Path) -> None:
    Outer(inner=Holder(x=torch.zeros(1), ys={})).save(tmp_path)
    assert isinstance(Outer.load(tmp_path).inner, Holder)


def test_tagged_rejects_non_spec_classes() -> None:
    with pytest.raises(ValidationError, match="not a Spec"):
        Outer.model_validate({"inner": {"type": "pathlib.Path"}})
