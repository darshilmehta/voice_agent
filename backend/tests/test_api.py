from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import create_app
from app.providers.registry import build_container

from .conftest import CLOUD_SECRETS, make_model_assets, mock_http


def _client(settings, http, **kw) -> TestClient:
    return TestClient(create_app(settings, build_container(settings, http=http, **kw), preload_models=False))


def _by_capability(body: dict) -> dict[str, dict]:
    return {p["capability"]: p for p in body["providers"]}


def test_health_all_ok(load_local, tmp_path):
    make_model_assets(tmp_path)
    s = load_local()
    with _client(s, mock_http(ollama_models=["qwen3:4b-instruct", "qwen3:8b"])) as client:
        body = client.get("/health").json()
    providers = _by_capability(body)
    assert body["status"] == "ok", {k: v for k, v in providers.items() if v["status"] not in ("ok", "disabled")}
    assert body["profile"] == "local" and body["strict_offline"] is True
    assert providers["web_search"]["status"] == "disabled"
    assert providers["llm"]["detail"] == "models ready: qwen3:4b-instruct"
    assert "will be created on first ingest" in providers["vector_store"]["detail"]
    assert (tmp_path / "data/sqlite/app.db").exists()  # created at startup
    assert (tmp_path / "data/uploads").is_dir()


def test_health_reports_missing_dependencies(load_local):
    s = load_local()  # no model files, Ollama and Qdrant unreachable
    with _client(s, mock_http(ollama_models=None, qdrant=False)) as client:
        body = client.get("/health").json()
    providers = _by_capability(body)
    assert body["status"] == "degraded"
    assert providers["llm"]["status"] == "down" and "unreachable" in providers["llm"]["detail"]
    assert providers["vector_store"]["status"] == "down"
    assert providers["embeddings"]["status"] == "down" and "not downloaded" in providers["embeddings"]["detail"]
    assert providers["ingestion"]["status"] == "down"
    assert providers["metadata_db"]["status"] == "ok"  # local, always available


def test_health_flags_models_not_pulled(load_local, tmp_path):
    make_model_assets(tmp_path)
    s = load_local(LLM__ROUTER_MODEL="qwen3:8b")
    with _client(s, mock_http(ollama_models=["qwen3:4b-instruct"])) as client:
        llm = _by_capability(client.get("/health").json())["llm"]
    assert llm["status"] == "degraded"
    assert "qwen3:8b" in llm["detail"]


def test_health_survives_a_crashing_check(load_local, tmp_path, monkeypatch):
    make_model_assets(tmp_path)
    s = load_local()
    container = build_container(s, http=mock_http(ollama_models=["qwen3:4b-instruct"]))

    async def boom():
        raise RuntimeError("kaboom")

    monkeypatch.setattr(container["tts"], "health", boom)
    with TestClient(create_app(s, container, preload_models=False)) as client:
        tts = _by_capability(client.get("/health").json())["tts"]
    assert tts["status"] == "down" and "kaboom" in tts["detail"]


def test_public_config_local(load_local):
    s = load_local()
    with _client(s, mock_http()) as client:
        body = client.get("/api/config/public").json()
    assert body["profile"] == "local"
    assert body["client"] == {"app_title": "Docent", "languages": ["en", "hi"], "default_language": "en"}
    assert body["features"] == {"voice": True, "web_search": False, "debug_panel": True}
    assert body["auth"]["provider"] == "none"
    assert ".pdf" in body["limits"]["allowed_extensions"]


def test_public_config_never_leaks_secrets(cloud_settings):
    with _client(cloud_settings, mock_http(), allow_placeholders=True) as client:
        raw = client.get("/api/config/public").text
    for secret in CLOUD_SECRETS.values():
        assert secret not in raw
    body = json.loads(raw)
    assert body["auth"]["client_id"] == "gibberlink-web"
    assert body["features"]["debug_panel"] is False


def test_cors_allows_configured_origin_only(load_local):
    s = load_local()
    with _client(s, mock_http()) as client:
        ok = client.get("/api/config/public", headers={"Origin": "http://localhost:3000"})
        other = client.get("/api/config/public", headers={"Origin": "https://evil.example"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert "access-control-allow-origin" not in other.headers
