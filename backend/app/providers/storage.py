"""Metadata database and object (file) store providers (docs/DESIGN.md §5, §6)."""

from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path

from .base import HealthStatus, PlaceholderProvider, Provider, ProviderHealth
from .registry import register


class MetadataDB(Provider):
    capability = "metadata_db"


@register
class SqliteDB(MetadataDB):
    name = "sqlite"

    @property
    def db_path(self) -> Path:
        url: str = self.config.url  # type: ignore[attr-defined]
        if ":///" not in url:
            raise ValueError(f"sqlite url must look like sqlite+aiosqlite:///path/to/app.db, got {url!r}")
        return self.ctx.settings.path(url.split(":///", 1)[1])

    async def start(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def health(self) -> ProviderHealth:
        path = self.db_path

        def ping() -> None:
            with sqlite3.connect(path, timeout=2) as conn:
                conn.execute("SELECT 1")

        try:
            await asyncio.to_thread(ping)
        except sqlite3.Error as e:
            return self._health(HealthStatus.DOWN, f"{path}: {e}")
        return self._health(HealthStatus.OK, f"{_rel(path, self.ctx.settings.root_dir)}")


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
