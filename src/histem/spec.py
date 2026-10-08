from pydantic import BaseModel, ConfigDict


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class FrozenSpec(Spec):
    model_config = ConfigDict(frozen=True)
