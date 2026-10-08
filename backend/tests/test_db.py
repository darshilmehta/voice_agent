"""Database layer: URL resolution, migrations, the sqlite provider, column types."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta, timezone

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Connection, inspect, select, text
from sqlalchemy.exc import IntegrityError

from app.db import migrate
from app.db import models as orm
from app.db.engine import create_engine, database_url
from app.db.types import new_id
from app.providers.registry import build_container

TABLES = {
    "projects",
    "documents",
    "document_versions",
    "ingestion_jobs",
    "chats",
    "messages",
    "chat_summaries",
    "document_tables",
}


@pytest.fixture
def url(load_local, tmp_path):
    (tmp_path / "fresh").mkdir()
    return database_url(load_local(METADATA_DB__URL=f"sqlite+aiosqlite:///{tmp_path}/fresh/app.db"))


def test_sqlite_url_resolves_against_project_root(load_local, tmp_path):
    url = database_url(load_local())
    assert url.drivername == "sqlite+aiosqlite"
    assert url.database == str((tmp_path / "data/sqlite/app.db").resolve())


def test_sync_sqlite_url_gets_the_async_driver(load_local, tmp_path):
    url = database_url(load_local(METADATA_DB__URL="sqlite:///db/x.db"))
    assert url.drivername == "sqlite+aiosqlite"
    assert url.database == str((tmp_path / "db/x.db").resolve())


@pytest.mark.parametrize("bad", ["sqlite+aiosqlite://", "sqlite+aiosqlite:///:memory:", "not a url"])
def test_unusable_urls_are_rejected(load_local, bad):
    with pytest.raises(ValueError, match="url"):
        database_url(load_local(METADATA_DB__URL=bad))


def test_postgres_url_is_used_as_is(cloud_settings):
    url = database_url(cloud_settings)
    assert url.drivername == "postgresql+asyncpg"
    assert url.host == "db.example.com"


async def test_migrations_build_every_table_on_a_fresh_database(url):
    await migrate.upgrade(url)
    engine = create_engine(url)
    try:
        async with engine.connect() as conn:
            tables = await conn.run_sync(lambda c: set(inspect(c).get_table_names()))
            assert tables == TABLES | {"alembic_version"}
            assert await migrate.current_revision(conn) == migrate.head_revision()
    finally:
        await engine.dispose()


async def test_migrations_match_the_models(url):
    """Fails when a model changes without a migration (run: uv run alembic revision --autogenerate)."""
    await migrate.upgrade(url)

    def diff(conn: Connection) -> list:
        return compare_metadata(MigrationContext.configure(conn, opts={"compare_type": True}), orm.Base.metadata)

    engine = create_engine(url)
    try:
        async with engine.connect() as conn:
            assert await conn.run_sync(diff) == []
    finally:
        await engine.dispose()


async def test_upgrade_is_idempotent_and_downgrade_drops_everything(url, caplog):
    caplog.set_level(logging.INFO)
    await migrate.upgrade(url)
    assert [r.getMessage().split(" (")[0] for r in caplog.records] == [
        f"metadata db schema empty → {migrate.head_revision()}"
    ]
    caplog.clear()
    await migrate.upgrade(url)  # every startup runs this: nothing to do, nothing logged
    assert caplog.records == []
    await migrate.downgrade(url, "base")
    engine = create_engine(url)
    try:
        async with engine.connect() as conn:
            tables = await conn.run_sync(lambda c: set(inspect(c).get_table_names()))
            assert tables == {"alembic_version"}
            assert await migrate.current_revision(conn) is None
    finally:
        await engine.dispose()


async def test_provider_migrates_on_start_and_reports_the_schema(db, tmp_path):
    assert (tmp_path / "data/sqlite/app.db").is_file()
    health = await db.health()
    assert health.status == "ok"
    assert health.detail == f"data/sqlite/app.db, schema {migrate.head_revision()}"


async def test_provider_before_start_and_after_close(load_local):
    container = build_container(load_local())
    db = container["metadata_db"]
    assert (await db.health()).status == "down"
    with pytest.raises(RuntimeError, match="not started"):
        async with db.session():
            pass
    await db.start()
    await db.close()
    assert "not started" in (await db.health()).detail
    await container.http.aclose()


async def test_connections_enforce_foreign_keys(db):
    with pytest.raises(IntegrityError, match="FOREIGN KEY"):
        async with db.session() as s:
            s.add(orm.Chat(project_id="prj_missing", title="orphan"))
    async with db.session() as s:
        assert await s.scalar(text("PRAGMA foreign_keys")) == 1
        assert await s.scalar(text("PRAGMA journal_mode")) == "wal"


async def test_session_commits_or_rolls_back(db):
    async with db.session() as s:
        s.add(orm.Project(id="prj_kept", name="kept"))
    with pytest.raises(RuntimeError):
        async with db.session() as s:
            s.add(orm.Project(id="prj_lost", name="lost"))
            await s.flush()
            raise RuntimeError("boom")
    async with db.session() as s:
        assert (await s.scalars(select(orm.Project.id))).all() == ["prj_kept"]


async def test_check_constraints_guard_enumerations(db):
    async with db.session() as s:
        s.add(orm.Project(id="prj_1", name="p"))
        await s.flush()  # no ORM relationships, so parents are flushed before children explicitly
        s.add(orm.Chat(id="cht_1", project_id="prj_1", title="c"))
    with pytest.raises(IntegrityError, match="ck_messages_role"):
        async with db.session() as s:
            s.add(orm.Message(chat_id="cht_1", seq=1, role="robot", text="hi"))


async def test_datetimes_come_back_timezone_aware_utc(db):
    ist = timezone(timedelta(hours=5, minutes=30))
    created = datetime(2026, 10, 8, 15, 0, tzinfo=ist)
    async with db.session() as s:
        s.add(orm.Project(id="prj_tz", name="tz", created_at=created, updated_at=created))
    async with db.session() as s:
        row = await s.get(orm.Project, "prj_tz")
        assert row is not None
        assert row.created_at == created
        assert row.created_at.tzinfo is UTC
        assert row.created_at.hour == 9 and row.created_at.minute == 30


async def test_naive_datetimes_are_refused(db):
    with pytest.raises(Exception, match="naive datetime"):
        async with db.session() as s:
            s.add(orm.Project(name="naive", created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 1)))


def test_postgres_placeholder_has_no_session(cloud_settings):
    db = build_container(cloud_settings, allow_placeholders=True)["metadata_db"]
    with pytest.raises(NotImplementedError):
        db.session()


def test_ids_are_prefixed_unique_and_time_ordered():
    first = new_id("prj")
    ids = [new_id("prj") for _ in range(1000)]
    assert re.fullmatch(r"prj_[0-9a-hjkmnp-tv-z]{26}", first)
    assert len(set(ids)) == 1000
    assert first[:14] <= min(ids)[:14]  # the timestamp part never goes backwards
