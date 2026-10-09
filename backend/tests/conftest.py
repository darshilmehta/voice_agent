from __future__ import annotations

import hashlib
import json
import os
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.db import models as orm
from app.main import create_app
from app.providers import models
from app.providers.base import ProviderContext
from app.providers.registry import build_container
from app.providers.storage import MetadataDB, SqliteDB
from app.settings import PROJECT_ROOT, Settings, load_settings

from .fakes import (
    FakeEmbedder,
    FakeLLM,
    FakeReranker,
    FakeStore,
    FakeSTT,
    FakeTTS,
    FakeVAD,
    FakeWebSearch,
    TextParser,
    keyword_scorer,
)

LOCAL_CONFIG = PROJECT_ROOT / "config/local.config.json"
CLOUD_CONFIG = PROJECT_ROOT / "config/cloud.config.json"
DOCKER_CONFIG = PROJECT_ROOT / "config/docker.config.json"

# Values for every ${VAR} in cloud.config.json. Distinctive so tests can assert they never leak.
CLOUD_SECRETS = {
    name: f"leak-canary-{name.lower()}"
    for name in (
        "LLM_API_KEY",
        "QDRANT_API_KEY",
        "DB_PASSWORD",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "REDIS_PASSWORD",
        "TURN_PASSWORD",
        "WEB_SEARCH_API_KEY",
    )
}


@pytest.fixture(autouse=True)
def _isolate_environ() -> Iterator[None]:
    """The app writes HF_* variables at startup; keep tests from leaking them into each other."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture(autouse=True)
def _apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    """Device checks behave as on the target machine regardless of where tests run."""
    monkeypatch.setattr(models, "APPLE_SILICON", True)


@pytest.fixture(autouse=True)
def _ml_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Health checks behave as if the optional ``ml`` group were installed (CI syncs without it)."""
    monkeypatch.setattr(models, "missing_modules", lambda names: [])


@pytest.fixture
def load_local(tmp_path: Path) -> Callable[..., Settings]:
    """Load config/local.config.json with the project root moved to tmp_path, plus env overrides."""

    def load(**env: str) -> Settings:
        return load_settings(LOCAL_CONFIG, {"APP_ROOT_DIR": str(tmp_path), **env})

    return load


@pytest.fixture
def write_config(tmp_path: Path) -> Callable[..., Path]:
    """Copy a config file with top-level keys changed. Top-level switches (profile, strict_offline,
    strict_offline_exceptions) are deliberately file-only: env overrides need SECTION__KEY names."""
    counter = iter(range(1000))

    def write(base: Path, **changes: object) -> Path:
        raw = json.loads(base.read_text())
        raw.update(changes)
        out = tmp_path / f"config-{next(counter)}.json"
        out.write_text(json.dumps(raw))
        return out

    return write


@pytest.fixture
def cloud_settings(tmp_path: Path) -> Settings:
    return load_settings(CLOUD_CONFIG, {"APP_ROOT_DIR": str(tmp_path), **CLOUD_SECRETS})


def make_model_assets(root: Path) -> None:
    """Create the on-disk layout of every local model the default config uses (empty files are enough)."""
    hub = root / "data/models/huggingface/hub"
    for repo in ("BAAI/bge-m3", "BAAI/bge-reranker-v2-m3", "mlx-community/whisper-small-mlx", "hexgrad/Kokoro-82M"):
        (hub / f"models--{repo.replace('/', '--')}" / "snapshots" / "abc123").mkdir(parents=True)
    voices = hub / "models--hexgrad--Kokoro-82M/snapshots/abc123/voices"
    voices.mkdir()
    for v in ("af_heart", "hf_alpha"):
        (voices / f"{v}.pt").touch()
    docling = root / "data/models/docling"
    for d in ("docling-project--docling-layout-heron", "docling-project--docling-models", "RapidOcr"):
        (docling / d).mkdir(parents=True)


def mock_http(*, ollama_models: list[str] | None = None, qdrant: bool = True) -> httpx.AsyncClient:
    """An HTTP client that answers like local Ollama and Qdrant (or refuses connections when absent)."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/tags":
            if ollama_models is None:
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(200, json={"models": [{"name": m} for m in ollama_models]})
        if path == "/readyz" or path.startswith("/collections/"):
            if not qdrant:
                raise httpx.ConnectError("connection refused", request=request)
            if path == "/readyz":
                return httpx.Response(200, text="all shards are ready")
            return httpx.Response(200, json={"result": {"exists": False}})
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class Clock:
    """A deterministic clock for services: every call is one second after the previous one."""

    def __init__(self, start: datetime = datetime(2026, 10, 8, 9, 0, tzinfo=UTC)) -> None:
        self.current = start

    def __call__(self) -> datetime:
        self.current += timedelta(seconds=1)
        return self.current


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
async def db(load_local) -> AsyncIterator[SqliteDB]:
    """The real sqlite provider on a fresh database under tmp_path, migrated to head."""
    settings = load_local()
    async with mock_http() as http:
        provider = SqliteDB(settings.metadata_db, ProviderContext(settings=settings, http=http))
        await provider.start()
        try:
            yield provider
        finally:
            await provider.close()


async def add_document(db: MetadataDB, project_id: str, filename: str = "annual_report.pdf", **fields: Any) -> str:
    """Insert a document with one version and one ingestion job, as the ingestion pipeline will. Returns its id."""
    async with db.session() as s:
        doc = orm.Document(
            project_id=project_id,
            filename=filename,
            mime="application/pdf",
            size_bytes=1024,
            sha256=hashlib.sha256(filename.encode()).hexdigest(),
            **fields,
        )
        s.add(doc)
        await s.flush()  # no ORM relationships: parents are flushed before children explicitly
        s.add(
            orm.DocumentVersion(
                document_id=doc.id,
                version=1,
                filename=filename,
                mime=doc.mime,
                size_bytes=doc.size_bytes,
                sha256=doc.sha256,
                storage_key=f"{project_id}/{doc.id}/v1/{filename}",
            )
        )
        s.add(orm.IngestionJob(document_id=doc.id, version=1))
        return doc.id


async def count_rows(db: MetadataDB, model: type[orm.Base]) -> int:
    async with db.session() as s:
        return await s.scalar(select(func.count()).select_from(model)) or 0


@pytest.fixture
def api(make_app) -> Iterator[TestClient]:
    """The app on a fresh database under tmp_path, with every external service faked. The Qdrant client opens its
    own connections (not the mocked HTTP client), so without the fake vector store a test would silently use a real
    Qdrant when one happens to run locally — and fail in CI, where none does."""
    with make_app() as client:
        yield client


@dataclass
class Fakes:
    """The providers an upload, a chat turn or a voice session uses, replaced by in-memory fakes (no ML, Qdrant or
    Ollama)."""

    parser: TextParser
    embedder: FakeEmbedder
    reranker: FakeReranker
    store: FakeStore
    llm: FakeLLM
    vad: FakeVAD = field(default_factory=FakeVAD)
    stt: FakeSTT = field(default_factory=FakeSTT)
    tts: FakeTTS = field(default_factory=FakeTTS)
    web: FakeWebSearch | None = None  # None: the configured provider (SearXNG, turned off) stays

    def install(self, container: Any) -> Any:
        self.parser.chunking_version = container.settings.ingestion.chunking.version  # as the real parser does
        container.providers.update(
            ingestion=self.parser,
            embeddings=self.embedder,
            reranker=self.reranker,
            vector_store=self.store,
            llm=self.llm,
            vad=self.vad,
            stt=self.stt,
            tts=self.tts,
        )
        if self.web is not None:
            container.providers["web_search"] = self.web
        return container


@pytest.fixture
def fakes() -> Fakes:
    return Fakes(TextParser(), FakeEmbedder(), FakeReranker(keyword_scorer), FakeStore(search_points=True), FakeLLM())


@pytest.fixture
def make_app(load_local, fakes) -> Callable[..., TestClient]:
    """``with make_app(**env) as api``: the app with the fakes installed, on the tmp_path database and uploads.

    Automatic chat titles are off unless ``auto_titles=True``: their background model call would otherwise show up in
    ``fakes.llm.calls`` of every test that counts the LLM calls of a turn."""

    def make(fakes_: Fakes | None = None, *, auto_titles: bool = False, **env: str) -> TestClient:
        settings = load_local(**env)
        container = (fakes_ or fakes).install(build_container(settings, http=mock_http()))
        return TestClient(create_app(settings, container, auto_titles=auto_titles))

    return make


@pytest.fixture
def app(make_app) -> Iterator[TestClient]:
    with make_app() as client:
        yield client
