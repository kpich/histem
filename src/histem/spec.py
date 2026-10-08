"""Base classes for validated configuration and value objects.

Hot-path array containers (`Population`, `Inputs`) are rebuilt every simulation step
and stay plain dataclasses; everything describing a model, experiment, or run is a
`Spec`, so construction is validated.
"""

from pydantic import BaseModel, ConfigDict


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class FrozenSpec(Spec):
    model_config = ConfigDict(frozen=True)
