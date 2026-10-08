"""Alembic environment.

Two ways in:

- from the app (``app.db.migrate``): a connection is passed in ``config.attributes["connection"]``;
- from the CLI (``uv run alembic …`` in ``backend/``): the URL comes from the app config (``APP_CONFIG_FILE``,
  ``METADATA_DB__URL`` overrides it), resolved exactly as the app resolves it.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from alembic import context
from sqlalchemy import Connection, pool
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.engine import database_url, is_sqlite
from app.db.models import Base
from app.db.types import UTCDateTime
from app.settings import load_settings

config = context.config
target_metadata = Base.metadata


def render_item(type_: str, obj: Any, autogen_context: Any) -> str | bool:
    """Render app column types as plain SQLAlchemy types, so migrations never import app code that may change."""
    if type_ != "type":
        return False
    if isinstance(obj, UTCDateTime):
        return "sa.DateTime(timezone=True)"
    if isinstance(obj, sa.JSON):  # app.db.types.JSONType: JSON, JSONB on Postgres
        autogen_context.imports.add("from sqlalchemy.dialects import postgresql")
        return 'sa.JSON().with_variant(postgresql.JSONB(), "postgresql")'
    return False


def configure(**kwargs: Any) -> None:
    context.configure(
        target_metadata=target_metadata,
        render_item=render_item,
        compare_type=True,
        **kwargs,
    )


def cli_url() -> URL:
    url = database_url(load_settings())
    if is_sqlite(url) and url.database:
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)
    return url


def run_migrations(connection: Connection) -> None:
    # SQLite can't ALTER most things; batch mode recreates the table instead.
    configure(connection=connection, render_as_batch=connection.dialect.name == "sqlite")
    with context.begin_transaction():
        context.run_migrations()


async def run_cli_migrations() -> None:
    engine = create_async_engine(cli_url(), poolclass=pool.NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():  # alembic upgrade head --sql
    url = cli_url()
    configure(url=url, literal_binds=True, render_as_batch=is_sqlite(url), dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()
elif (shared := config.attributes.get("connection")) is not None:
    run_migrations(shared)
else:
    if config.config_file_name is not None:
        fileConfig(config.config_file_name, disable_existing_loggers=False)
    asyncio.run(run_cli_migrations())
