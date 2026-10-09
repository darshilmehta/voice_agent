from __future__ import annotations

import pytest

from app.providers import load_all
from app.providers.base import PlaceholderProvider
from app.providers.registry import CAPABILITIES, build_container, registered
from app.settings import ConfigError

LOCAL_PROVIDERS = {
    "llm": "ollama",
    "embeddings": "bge_m3",
    "reranker": "bge_reranker",
    "vector_store": "qdrant",
    "metadata_db": "sqlite",
    "object_store": "filesystem",
    "ingestion": "docling",
    "stt": "mlx_whisper",
    "vad": "silero",
    "tts": "kokoro",
    "audio_transport": "websocket",
    "auth": "none",
    "job_queue": "in_process",
    "session_store": "in_memory",
    "event_bus": "in_process",
    "web_search": "searxng",
}

# What cloud.config.json selects that is still a placeholder (docs/DESIGN.md §6.4).
CLOUD_PLACEHOLDERS = {
    "llm": "openai_compatible",
    "metadata_db": "postgres",
    "object_store": "s3",
    "audio_transport": "webrtc",
    "auth": "oidc",
    "job_queue": "redis",
    "session_store": "redis",
    "event_bus": "redis",
    "web_search": "search_api",
}


def test_every_capability_has_an_implemented_provider():
    load_all()
    for capability in CAPABILITIES:
        implemented = [n for n, cls in registered(capability).items() if cls.implemented]
        assert implemented, f"{capability} has no implemented provider"


def test_local_config_wires_every_capability(load_local):
    c = build_container(load_local())
    assert {cap: p.name for cap, p in c.providers.items()} == LOCAL_PROVIDERS
    assert c.placeholders == []


def test_cloud_config_wires_with_placeholders(cloud_settings):
    c = build_container(cloud_settings, allow_placeholders=True)
    placeholders = {cap: c[cap].name for cap in c.placeholders}
    assert placeholders == CLOUD_PLACEHOLDERS
    # the self-hosted parts are real providers, configured for a GPU server
    assert c["embeddings"].name == "bge_m3" and not isinstance(c["embeddings"], PlaceholderProvider)
    assert c["stt"].name == "faster_whisper"
    assert c["vector_store"].name == "qdrant"


def test_app_refuses_placeholders(cloud_settings):
    with pytest.raises(ConfigError) as err:
        build_container(cloud_settings)
    for cap, name in CLOUD_PLACEHOLDERS.items():
        assert f"{cap}={name}" in str(err.value)


def test_placeholder_methods_raise_not_implemented(cloud_settings):
    c = build_container(cloud_settings, allow_placeholders=True)
    with pytest.raises(NotImplementedError, match="placeholder"):
        c["object_store"].put  # noqa: B018  (attribute access is the call under test)


@pytest.mark.parametrize("method", ["submit", "on_idle", "join"])
def test_the_redis_job_queue_placeholder_has_the_lane_aware_interface(cloud_settings, method):
    """The cloud placeholder stands in for every JobQueue method, including the ones that take a lane."""
    queue = build_container(cloud_settings, allow_placeholders=True)["job_queue"]
    assert queue.name == "redis" and isinstance(queue, PlaceholderProvider)
    with pytest.raises(NotImplementedError, match=rf"placeholder: '{method}' is not implemented"):
        getattr(queue, method)


def test_unknown_provider_lists_alternatives(load_local):
    with pytest.raises(
        ConfigError, match=r"stt: unknown provider 'whisper_cpp' \(available: faster_whisper, mlx_whisper\)"
    ):
        build_container(load_local(STT__PROVIDER="whisper_cpp"))
