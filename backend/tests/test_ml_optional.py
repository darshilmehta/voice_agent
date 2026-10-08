"""The ML stack is optional: nothing heavy is imported until a model is used, and its absence is reported clearly."""

from __future__ import annotations

import asyncio
import subprocess
import sys

import pytest

from app.providers import models
from app.providers.ingestion import DoclingParser, IngestionError
from app.providers.models import ML_INSTALL_HINT, ModelUnavailableError
from app.providers.registry import build_container
from app.providers.retrieval import BgeM3Embedder, BgeReranker
from app.settings import PROJECT_ROOT

from .conftest import make_model_assets

HEAVY = ("torch", "docling", "docling_core", "FlagEmbedding", "sentence_transformers", "transformers", "onnxruntime")


def test_importing_the_app_loads_no_ml_library():
    code = (
        "import sys, app.main, app.services.ingestion, app.services.retrieval\n"
        "from app.providers import load_all; load_all()\n"
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT / "backend", capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""


@pytest.fixture
def providers(load_local, tmp_path):
    make_model_assets(tmp_path)
    c = build_container(load_local())
    return c["embeddings"], c["reranker"], c["ingestion"]


def ml_missing(monkeypatch):
    monkeypatch.setattr(models, "missing_modules", lambda names: list(names))


def test_models_are_not_loaded_at_construction_or_health(providers):
    embedder, reranker, parser = providers
    assert isinstance(embedder, BgeM3Embedder) and isinstance(reranker, BgeReranker)
    assert isinstance(parser, DoclingParser)
    for p in providers:
        assert asyncio.run(p.health()).status == "ok"
        assert p.loaded is False


def test_empty_input_never_loads_a_model(providers):
    embedder, reranker, _ = providers
    assert asyncio.run(embedder.embed([])) == []
    assert asyncio.run(reranker.score("q", [])) == []
    assert not embedder.loaded and not reranker.loaded


def test_missing_ml_group_is_a_clear_error(providers, monkeypatch, tmp_path):
    embedder, reranker, parser = providers
    ml_missing(monkeypatch)
    with pytest.raises(ModelUnavailableError, match="uv sync --group ml"):
        asyncio.run(embedder.embed(["text"]))
    with pytest.raises(ModelUnavailableError, match="uv sync --group ml"):
        asyncio.run(reranker.score("q", ["p"]))
    doc = tmp_path / "notes.md"
    doc.write_text("# Notes\n\nHello.")
    with pytest.raises(ModelUnavailableError, match="uv sync --group ml"):
        asyncio.run(parser.parse(doc))


def test_missing_ml_group_degrades_health(providers, monkeypatch):
    ml_missing(monkeypatch)
    for p in providers:
        h = asyncio.run(p.health())
        assert h.status == "degraded" and ML_INSTALL_HINT in h.detail


def test_missing_model_files_are_reported_before_any_import(load_local):
    embedder = build_container(load_local())["embeddings"]  # no model assets in the temp project root
    with pytest.raises(ModelUnavailableError, match="not downloaded"):
        asyncio.run(embedder.embed(["text"]))
    assert not embedder.loaded


def test_unload_without_a_model_is_a_no_op(providers):
    for p in providers:
        asyncio.run(p.close())


@pytest.mark.parametrize(("name", "match"), [("virus.exe", "not allowed"), ("page.html", "not allowed")])
def test_parser_rejects_disallowed_files_before_loading_anything(providers, tmp_path, name, match):
    _, _, parser = providers
    f = tmp_path / name
    f.write_text("x")
    with pytest.raises(IngestionError, match=match):
        asyncio.run(parser.parse(f))
    assert not parser.loaded


def test_parser_reports_missing_files(providers, tmp_path):
    _, _, parser = providers
    with pytest.raises(IngestionError, match="not found"):
        asyncio.run(parser.parse(tmp_path / "gone.pdf"))
