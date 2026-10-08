"""Shared pieces of the service layer: errors, the "not given" marker for partial updates, the base class."""

from __future__ import annotations

import enum
from collections.abc import Callable
from datetime import datetime
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import Base
from ..db.types import utcnow
from ..providers.storage import MetadataDB


class ServiceError(Exception):
    """An error caused by the request, not by the server. The API turns these into 4xx responses."""


class NotFound(ServiceError):
    def __init__(self, kind: str, id: str) -> None:
        super().__init__(f"{kind} {id!r} not found")
        self.kind = kind
        self.id = id


class InvalidInput(ServiceError):
    """Well-formed but unacceptable input, e.g. a document scope naming documents of another project."""


class Unavailable(Exception):
    """A dependency the operation needs is down (e.g. the vector store during a delete). The API answers 503; the
    operation changed nothing and can be retried."""


class _Unset(enum.Enum):
    UNSET = "UNSET"


UNSET = _Unset.UNSET
"""Default of every partial-update argument: "leave as is", as opposed to None, which clears the value."""

type Maybe[T] = T | Literal[_Unset.UNSET]


class Service:
    def __init__(self, db: MetadataDB, *, clock: Callable[[], datetime] = utcnow) -> None:
        self.db = db
        self.now = clock


async def get_or_404[M: Base](session: AsyncSession, model: type[M], id: str, kind: str) -> M:
    row = await session.get(model, id)
    if row is None:
        raise NotFound(kind, id)
    return row


def clean_text(value: str, field: str, max_length: int) -> str:
    """Strip surrounding whitespace; reject empty or over-long values."""
    value = value.strip()
    if not value:
        raise InvalidInput(f"{field} must not be empty")
    if len(value) > max_length:
        raise InvalidInput(f"{field} is longer than {max_length} characters")
    return value


def blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip() or None
