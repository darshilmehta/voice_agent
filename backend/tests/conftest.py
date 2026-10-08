from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest

from app.providers import models
from app.settings import PROJECT_ROOT, Settings, load_settings

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
