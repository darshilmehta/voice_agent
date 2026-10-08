"""Column types and helpers that behave the same on SQLite and Postgres."""

from __future__ import annotations

import secrets
import time
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator

# Crockford base32 (no i, l, o, u), lowercase so ids read well in URLs and logs.
_ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    """A readable, globally unique id such as ``prj_01k7b6p1x8c2mz3q9t4w5v6r7s``.

    The 26 characters after the prefix use the ULID layout (48-bit millisecond timestamp + 80 random bits), so
    ids sort roughly by creation time and work unchanged in any database or in Qdrant payloads.
    """
    value = (time.time_ns() // 1_000_000) << 80 | secrets.randbits(80)
    chars = []
    for _ in range(26):
        value, digit = divmod(value, 32)
        chars.append(_ALPHABET[digit])
    return f"{prefix}_{''.join(reversed(chars))}"


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC datetimes on every backend.

    Postgres stores ``timestamptz``. SQLite has no time zone type: values are stored as naive UTC (whose text form
    sorts chronologically) and marked UTC again when read.
    """

    impl = DateTime
    cache_ok = True

    def __init__(self) -> None:
        super().__init__(timezone=True)

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(f"naive datetime {value!r}: use timezone-aware UTC values (app.db.types.utcnow)")
        value = value.astimezone(UTC)
        return value.replace(tzinfo=None) if dialect.name == "sqlite" else value

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


# JSON on SQLite (stored as text), JSONB on Postgres. Python None is stored as SQL NULL, not the JSON literal null,
# so "IS NULL" queries work (document_scope: null means "all documents").
JSONType: Any = JSON(none_as_null=True).with_variant(postgresql.JSONB(none_as_null=True), "postgresql")
