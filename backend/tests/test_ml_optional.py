"""The ML stack is optional: nothing heavy is imported until a model is used, and its absence is reported clearly."""

from __future__ import annotations

import asyncio
import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from app.providers import models
from app.providers.ingestion import DoclingParser, IngestionError
from app.providers.models import ML_INSTALL_HINT, ModelUnavailableError
from app.providers.registry import build_container
from app.providers.retrieval import BgeM3Embedder, BgeReranker
from app.providers.speech import KokoroTTS, MlxWhisper, SileroVAD, pick_language, silero_model_path
from app.services.preload import ModelPreloader
from app.settings import PROJECT_ROOT

from .conftest import make_model_assets, mock_http

HEAVY = (
    "torch",
    "docling",
    "docling_core",
    "FlagEmbedding",
    "sentence_transformers",
    "transformers",
    "onnxruntime",
    "mlx",
    "mlx_whisper",
    "faster_whisper",
    "ctranslate2",
    "kokoro",
    "misaki",
    "spacy",
    "silero_vad",
)


def test_importing_the_app_loads_no_ml_library():
    code = (
        "import sys, app.main, app.services.ingestion, app.services.retrieval, app.services.voice\n"
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


# ------------------------------------------------------------------ speech


@pytest.fixture
def speech(load_local, tmp_path):
    make_model_assets(tmp_path)
    c = build_container(load_local())
    return c["vad"], c["stt"], c["tts"]


def test_speech_models_are_not_loaded_at_construction_or_health(speech):
    vad, stt, tts = speech
    assert isinstance(vad, SileroVAD) and isinstance(stt, MlxWhisper) and isinstance(tts, KokoroTTS)
    for p in speech:
        h = asyncio.run(p.health())
        assert h.status == "ok", h
        assert "loads on first use" in h.detail
        assert p.loaded is False


def test_missing_ml_group_is_a_clear_error_for_speech(speech, monkeypatch):
    vad, stt, tts = speech
    ml_missing(monkeypatch)
    with pytest.raises(ModelUnavailableError, match="uv sync --group ml"):
        asyncio.run(stt.transcribe(np.zeros(1600, dtype=np.float32), ["en"]))
    with pytest.raises(ModelUnavailableError, match="uv sync --group ml"):
        asyncio.run(tts.synthesize("Hello.", "en"))
    with pytest.raises(ModelUnavailableError, match="uv sync --group ml"):
        asyncio.run(vad.open_stream()(np.zeros((1, 512), dtype=np.float32)))
    for p in speech:
        h = asyncio.run(p.health())
        assert h.status == "degraded" and ML_INSTALL_HINT in h.detail
        assert not p.loaded


def test_tts_refuses_a_language_without_a_voice_before_loading(speech):
    _, _, tts = speech
    with pytest.raises(ValueError, match="no Kokoro voice"):
        asyncio.run(tts.synthesize("Bonjour", "fr"))  # type: ignore[arg-type]
    assert not tts.loaded


def test_whisper_language_detection_is_restricted_to_the_configured_languages():
    # Whisper alone may hear Hindi as Urdu; only the allowed languages compete
    assert pick_language({"ur": 0.6, "hi": 0.3, "en": 0.1}, ["en", "hi"]) == ("hi", 0.3)
    assert pick_language({"en": 0.9}, ["hi"]) == ("hi", 0.0)


def test_preload_reports_models_it_cannot_load_without_failing(load_local):
    async def go():
        container = build_container(load_local(), http=mock_http())  # no model files; Ollama answers 404
        preloader = ModelPreloader(container)
        assert preloader.report().state == "off"
        preloader.start()
        await preloader.wait()
        return preloader.report()

    report = asyncio.run(go())
    states = {m.capability: m for m in report.models}
    assert report.state == "degraded"
    for cap in ("stt", "tts", "embeddings", "reranker"):
        assert states[cap].state == "unavailable" and "not downloaded" in states[cap].detail
    assert states["llm"].state == "failed" and "404" in states["llm"].detail
    assert states["vad"].state in ("ready", "unavailable")  # the Silero model ships with the ml group


@pytest.mark.skipif(
    silero_model_path() is None or importlib.util.find_spec("onnxruntime") is None, reason="needs the ml group"
)
def test_real_silero_keeps_one_state_per_stream(speech):
    vad, _, _ = speech
    rng = np.random.default_rng(0)
    quiet = (rng.standard_normal((20, 512)) * 0.003).astype(np.float32)
    first, second = vad.open_stream(), vad.open_stream()
    a = asyncio.run(first(quiet))
    b = asyncio.run(second(quiet))
    assert len(a) == 20 and a == pytest.approx(b)  # same input, fresh state each: same output
    assert max(a) < 0.5  # quiet noise is not speech
    first.reset()
    assert asyncio.run(first(quiet)) == pytest.approx(a)
