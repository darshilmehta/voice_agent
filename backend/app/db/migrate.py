"""Run Alembic migrations from code. The ``sqlite`` metadata DB provider upgrades to head on every startup.

From the command line (``backend/alembic.ini``), against the database the app config points at:

    uv run alembic upgrade head
    uv run alembic revision --autogenerate --rev-id 0002 -m "add something"
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, inspect, pool, text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

log = logging.getLogger(__name__)


def alembic_config() -> Config:
    """An in-memory Alembic config (no ini file, so it never reconfigures the app's logging)."""
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("path_separator", "os")
    return cfg


def head_revision() -> str:
    head = ScriptDirectory.from_config(alembic_config()).get_current_head()
    if head is None:
        raise RuntimeError(f"no migrations found in {MIGRATIONS_DIR}")
    return head


async def upgrade(url: URL, revision: str = "head") -> None:
    """Migrate the database at ``url`` up to ``revision``."""
    await _run(url, "upgrade", revision)


async def downgrade(url: URL, revision: str) -> None:
    """Migrate the database at ``url`` down to ``revision`` (``"base"`` drops every table)."""
    await _run(url, "downgrade", revision)


async def current_revision(conn: AsyncConnection) -> str | None:
    """The schema revision recorded in the database; None before the first migration."""

    def read(sync_conn: Connection) -> str | None:
        # Plain SQL rather than Alembic's MigrationContext, which logs at INFO on every call (this runs in /health).
        if not inspect(sync_conn).has_table("alembic_version"):
            return None
        return sync_conn.execute(text("SELECT version_num FROM alembic_version")).scalar()

    return await conn.run_sync(read)


async def _run(url: URL, direction: str, revision: str) -> None:
    # A dedicated, unpooled connection: SQLite starts it with foreign keys off (the app engine turns them on).
    # That is what batch migrations need, since they recreate tables, and dropping a referenced table with
    # foreign keys on would cascade-delete its children. NullPool closes the connection afterwards, so the app
    # never reuses it.
    engine = create_async_engine(url, poolclass=pool.NullPool)
    try:
        async with engine.connect() as conn:
            before = await current_revision(conn)
            with _quiet_alembic():
                await conn.run_sync(_run_sync, direction, revision)
            await conn.commit()
            after = await current_revision(conn)
    finally:
        await engine.dispose()
    if after != before:
        log.info("metadata db schema %s → %s (%s)", before or "empty", after or "empty", url.render_as_string())


def _run_sync(connection: Connection, direction: str, revision: str) -> None:
    cfg = alembic_config()
    cfg.attributes["connection"] = connection  # env.py migrates on this connection instead of opening its own
    getattr(command, direction)(cfg, revision)


@contextmanager
def _quiet_alembic() -> Iterator[None]:
    """Alembic logs several INFO lines per run; the app logs one line itself when the schema changes."""
    logger = logging.getLogger("alembic")
    level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(level)
