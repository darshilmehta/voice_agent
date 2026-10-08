"""Metadata database and object (file) store providers (docs/DESIGN.md §5, §6)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from ..db import migrate
from ..db.engine import create_engine, database_url, is_sqlite
from .base import HealthStatus, PlaceholderProvider, Provider, ProviderContext, ProviderHealth
from .registry import register


class MetadataDB(Provider):
    """Projects, documents, chats and messages (docs/DESIGN.md §3.9). Business logic (``app/services``) depends on
    this interface only: it opens a unit of work with ``session()`` and uses the models in ``app.db.models``."""

    capability = "metadata_db"

    def session(self) -> AbstractAsyncContextManager[AsyncSession]:
        """A unit of work: ``async with db.session() as s: …`` commits when the block succeeds and rolls back
        if it raises."""
        raise NotImplementedError(f"metadata_db provider {self.name!r} does not implement session()")


@register
class SqliteDB(MetadataDB):
    """SQLite through SQLAlchemy (aiosqlite). ``start()`` creates the directory and migrates the schema to head."""

    name = "sqlite"

    def __init__(self, config: BaseModel, ctx: ProviderContext) -> None:
        super().__init__(config, ctx)
        self._engine: AsyncEngine | None = None
        self._sessions: async_sessionmaker[AsyncSession] | None = None

    @property
    def url(self) -> URL:
        url = database_url(self.ctx.settings)
        if not is_sqlite(url):
            raise ValueError(f"metadata_db provider 'sqlite' needs a sqlite+aiosqlite:/// url, got {url.drivername}")
        return url

    @property
    def db_path(self) -> Path:
        return Path(self.url.database or "")

    async def start(self) -> None:
        url = self.url
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        await migrate.upgrade(url)
        self._engine = create_engine(url)
        self._sessions = async_sessionmaker(self._engine, expire_on_commit=False)

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
        self._engine = self._sessions = None

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[AsyncSession]:
        if self._sessions is None:
            raise RuntimeError("metadata_db is not started")
        async with self._sessions() as session, session.begin():
            yield session

    def session(self) -> AbstractAsyncContextManager[AsyncSession]:
        return self._session()

    async def health(self) -> ProviderHealth:
        where = _rel(self.db_path, self.ctx.settings.root_dir)
        if self._engine is None:
            return self._health(HealthStatus.DOWN, f"{where}: not started")
        try:
            async with self._engine.connect() as conn:
                revision = await migrate.current_revision(conn)
        except SQLAlchemyError as e:
            return self._health(HealthStatus.DOWN, f"{where}: {e}")
        return self._health(HealthStatus.OK, f"{where}, schema {revision}")


@register
class PostgresDB(MetadataDB, PlaceholderProvider):
    name = "postgres"


class ObjectStore(Provider):
    capability = "object_store"


@register
class FilesystemStore(ObjectStore):
    name = "filesystem"

    @property
    def root(self) -> Path:
        root = self.config.root  # type: ignore[attr-defined]
        if not root:
            raise ValueError("object_store.root is required for the filesystem provider")
        return self.ctx.settings.path(root)

    async def start(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    async def health(self) -> ProviderHealth:
        root = self.root
        if not root.is_dir():
            return self._health(HealthStatus.DOWN, f"{root} does not exist")
        if not os.access(root, os.W_OK):
            return self._health(HealthStatus.DOWN, f"{root} is not writable")
        return self._health(HealthStatus.OK, f"{_rel(root, self.ctx.settings.root_dir)} writable")


@register
class S3Store(ObjectStore, PlaceholderProvider):
    name = "s3"


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
