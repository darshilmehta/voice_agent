"""Request-body building blocks shared by the routers."""

from __future__ import annotations

from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
LongText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=4000)]


class Body(BaseModel):
    """Request bodies reject unknown fields, so a typo is a 422 instead of a silent no-op."""

    model_config = ConfigDict(extra="forbid")


class Patch(Body):
    """A partial update: fields left out stay as they are; ``null`` clears a field where that makes sense.
    Fields in ``NOT_NULL`` can be left out but not set to null."""

    NOT_NULL: ClassVar[tuple[str, ...]] = ()

    @model_validator(mode="after")
    def _reject_nulls(self) -> Patch:
        nulls = [f for f in self.NOT_NULL if f in self.model_fields_set and getattr(self, f) is None]
        if nulls:
            raise ValueError(f"cannot be null: {', '.join(nulls)}")
        return self

    def changes(self) -> dict[str, Any]:
        """Only the fields the client sent, ready to pass to a service's ``update(**changes)``."""
        return {field: getattr(self, field) for field in self.model_fields_set}
