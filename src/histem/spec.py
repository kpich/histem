import importlib
from pathlib import Path
from typing import Annotated, Any, Self, TypeVar

import torch
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainSerializer,
    PlainValidator,
    SerializationInfo,
    ValidationInfo,
)

T = TypeVar("T")


class Spec(BaseModel):
    """Saved as config.json plus tensors.pt; tensors in the JSON are keys into it."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def save(self, path: Path) -> None:
        tensors: dict[str, torch.Tensor] = {}
        config = self.model_dump_json(indent=2, context={"tensors": tensors})
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(config)
        torch.save(tensors, path / "tensors.pt")

    @classmethod
    def load(cls, path: Path) -> Self:
        tensors = torch.load(path / "tensors.pt", weights_only=True)
        return cls.model_validate_json(
            (path / "config.json").read_text(), context={"tensors": tensors}
        )


class FrozenSpec(Spec):
    model_config = ConfigDict(frozen=True)


def _store(info: SerializationInfo | ValidationInfo) -> dict[str, torch.Tensor] | None:
    context = info.context or {}
    store: dict[str, torch.Tensor] | None = context.get("tensors")
    return store


def _dump_tensor(t: torch.Tensor, info: SerializationInfo) -> Any:
    store = _store(info)
    if store is None:
        return t.tolist() if info.mode == "json" else t
    key = str(len(store))
    store[key] = t.detach()
    return {"tensor": key}


def _load_tensor(v: Any, info: ValidationInfo) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v
    store = _store(info)
    if isinstance(v, dict) and store is not None:
        return store[v["tensor"]]
    raise ValueError(f"expected a tensor, got {type(v).__name__}")


Tensor = Annotated[
    torch.Tensor, PlainValidator(_load_tensor), PlainSerializer(_dump_tensor)
]


def _dump_tagged(v: Spec, info: SerializationInfo) -> Any:
    tag = f"{type(v).__module__}.{type(v).__name__}"
    return {"type": tag, **v.model_dump(mode=info.mode, context=info.context)}


def _load_tagged(v: Any, info: ValidationInfo) -> Any:
    if not isinstance(v, dict):
        return v
    v = dict(v)
    module, _, name = v.pop("type").rpartition(".")
    cls = getattr(importlib.import_module(module), name)
    if not (isinstance(cls, type) and issubclass(cls, Spec)):
        raise ValueError(f"{module}.{name} is not a Spec")
    return cls.model_validate(v, context=info.context)


# a Protocol-typed field, saved with the implementing class's import path
Tagged = Annotated[T, BeforeValidator(_load_tagged), PlainSerializer(_dump_tagged)]
