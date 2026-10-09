"""Integration tests: real models, real Docling, a running Qdrant. Opt-in.

    RUN_INTEGRATION=1 uv run --group ml pytest tests/integration -s

Environment (defaults suit the main checkout, where data/ holds the models and smoke fixtures):
    MODELS_ROOT   directory with huggingface/ (HF_HOME) and docling/ (artifacts)   default <repo>/data/models
    SMOKE_DOCS    directory with annual_report.pdf (written by scripts/smoke/05)  default <repo>/data/smoke/docs
    SMOKE_AUDIO   directory with the speech clips (scripts/smoke/09)          default <repo>/data/smoke/audio
    QDRANT_URL    a local Qdrant; a uniquely named collection is created and dropped default http://127.0.0.1:6333
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.offline import apply_runtime_env
from app.providers.registry import Container, build_container
from app.settings import PROJECT_ROOT, Settings, load_settings

LOCAL_CONFIG = PROJECT_ROOT / "config/local.config.json"
METRICS: dict[str, Any] = {}  # printed in the terminal summary


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("RUN_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="integration test: set RUN_INTEGRATION=1 (needs the ml group, models and Qdrant)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


def pytest_terminal_summary(terminalreporter: Any) -> None:
    if METRICS:
        terminalreporter.section("integration metrics")
        for key, value in METRICS.items():
            terminalreporter.write_line(f"{key}: {value}")


@pytest.fixture(scope="session")
def models_root() -> Path:
    root = Path(os.environ.get("MODELS_ROOT") or PROJECT_ROOT / "data/models")
    if not (root / "huggingface").is_dir() or not (root / "docling").is_dir():
        pytest.skip(f"models not found under {root} (set MODELS_ROOT)")
    return root


@pytest.fixture(scope="session")
def smoke_docs() -> Path:
    docs = Path(os.environ.get("SMOKE_DOCS") or PROJECT_ROOT / "data/smoke/docs")
    if not (docs / "annual_report.pdf").is_file():
        pytest.skip(f"annual_report.pdf not found in {docs} (run scripts/smoke/05_docling.py or set SMOKE_DOCS)")
    return docs


@pytest.fixture(scope="session")
def smoke_audio() -> Path:
    audio = Path(os.environ.get("SMOKE_AUDIO") or PROJECT_ROOT / "data/smoke/audio")
    if not (audio / "say_en_0.wav").is_file():
        pytest.skip(f"speech clips not found in {audio} (run scripts/smoke/09_kokoro.py or set SMOKE_AUDIO)")
    return audio


def require_chat_model(settings: Settings) -> None:
    """Skip unless Ollama is reachable and has the configured chat model."""
    try:
        pulled = {m["name"] for m in httpx.get(f"{settings.llm.base_url}/api/tags", timeout=2).json()["models"]}
    except httpx.HTTPError as e:
        pytest.skip(f"Ollama not reachable at {settings.llm.base_url}: {e}")
    if settings.llm.chat_model not in pulled:
        pytest.skip(f"{settings.llm.chat_model} not pulled in Ollama")


@pytest.fixture(scope="session")
def loop() -> Iterator[asyncio.AbstractEventLoop]:
    """One event loop for the session: the Qdrant client's connections belong to the loop that opened them."""
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def integration_settings(models_root: Path, root: Path) -> Settings:
    """Local config with the project root at ``root`` (fresh database and uploads), models from ``models_root`` and a
    new, uniquely named Qdrant collection (the caller drops it). Skips when Qdrant isn't reachable."""
    qdrant = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    try:
        httpx.get(f"{qdrant}/readyz", timeout=2).raise_for_status()
    except httpx.HTTPError as e:
        pytest.skip(f"Qdrant not reachable at {qdrant}: {e}")
    env = {
        **os.environ,  # lets EMBEDDINGS__DEVICE=cpu etc. through
        "APP_ROOT_DIR": str(root),
        "MODEL_CACHE__HF_HOME": str(models_root / "huggingface"),
        "INGESTION__ARTIFACTS_PATH": str(models_root / "docling"),
        "VECTOR_STORE__URL": qdrant,
        "VECTOR_STORE__COLLECTION": f"it_{uuid.uuid4().hex[:12]}",
    }
    s = load_settings(LOCAL_CONFIG, env)
    apply_runtime_env(s)  # HF_HOME + HF_HUB_OFFLINE=1 (strict_offline), as the app does at startup
    return s


@pytest.fixture(scope="session")
def settings(models_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return integration_settings(models_root, tmp_path_factory.mktemp("root"))


@pytest.fixture(scope="session")
def container(settings: Settings, loop: asyncio.AbstractEventLoop) -> Iterator[Container]:
    """Every provider once per session (models load on first use); collection dropped and models freed at the end."""
    c = build_container(settings)
    yield c
    if sys.platform != "win32":
        import resource

        scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB on Linux
        METRICS["peak RSS"] = f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 1e9:.2f} GB"
    store = c["vector_store"]
    loop.run_until_complete(store.drop_collection())  # type: ignore[attr-defined]
    loop.run_until_complete(c.close())
