"""Database URLs and engines.

``metadata_db.url`` is a SQLAlchemy URL. Relative SQLite paths resolve against the project root like every other
configured path (``sqlite+aiosqlite:///data/sqlite/app.db`` → ``<root>/data/sqlite/app.db``); other URLs
(``postgresql+asyncpg://…``) are used as they are.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ..settings import Settings

SQLITE_ASYNC_DRIVER = "sqlite+aiosqlite"


def database_url(settings: Settings) -> URL:
    """The configured metadata DB URL, ready for ``create_async_engine``."""
    raw = settings.metadata_db.url
    try:
        url = make_url(raw)
    except Exception as e:  # sqlalchemy raises ArgumentError; keep the message about the config value
        raise ValueError(f"metadata_db.url is not a database URL: {raw!r}") from e
    if url.get_backend_name() != "sqlite":
        return url
    if not url.database or url.database == ":memory:" or url.database.startswith("file:"):
        # An in-memory database would be a different, empty database on every pooled connection.
        raise ValueError(f"sqlite url must name a file, like sqlite+aiosqlite:///data/sqlite/app.db; got {raw!r}")
    return url.set(drivername=SQLITE_ASYNC_DRIVER, database=str(settings.path(url.database)))


def is_sqlite(url: URL) -> bool:
    return url.get_backend_name() == "sqlite"


def create_engine(url: URL, **kwargs: Any) -> AsyncEngine:
    """An async engine. On SQLite every connection enforces foreign keys (needed for ON DELETE CASCADE) and the
    database uses write-ahead logging, so readers don't block the writer."""
    engine = create_async_engine(url, **kwargs)
    if is_sqlite(url):
        event.listen(engine.sync_engine, "connect", _sqlite_pragmas)
    return engine


def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
    finally:
        cursor.close()
