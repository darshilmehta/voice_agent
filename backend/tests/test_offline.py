from __future__ import annotations

import pytest

from app.offline import apply_runtime_env, is_loopback_url, offline_violations
from app.providers.registry import build_container
from app.settings import ConfigError, load_settings

from .conftest import CLOUD_CONFIG, CLOUD_SECRETS, LOCAL_CONFIG


@pytest.mark.parametrize(
    ("url", "loopback"),
    [
        ("http://127.0.0.1:11434", True),
        ("http://localhost:8000", True),
        ("http://[::1]:6333", True),
        ("http://api.localhost", True),
        ("sqlite+aiosqlite:///data/sqlite/app.db", True),
        ("https://qdrant.example.com:6333", False),
        ("http://10.0.0.5:6333", False),  # private network is still the network
        ("postgresql+asyncpg://u:p@db.example.com:5432/x", False),
        ("stun:stun.example.com:3478", False),
    ],
)
def test_is_loopback_url(url, loopback):
    assert is_loopback_url(url) is loopback


def test_local_config_has_no_violations(load_local):
    assert offline_violations(load_local(), {}) == []


def test_remote_url_is_a_violation(load_local):
    s = load_local(VECTOR_STORE__URL="https://qdrant.example.com:6333")
    assert offline_violations(s, {}) == ["vector_store.url: https://qdrant.example.com:6333 is not a loopback address"]
    with pytest.raises(ConfigError, match="strict_offline"):
        build_container(s)


def test_disabled_web_search_is_allowed(load_local):
    build_container(load_local())  # searxng is remote, but disabled


def test_enabled_web_search_needs_an_exception(load_local, write_config, tmp_path):
    s = load_local(TOOLS__WEB_SEARCH__ENABLED="true")
    with pytest.raises(ConfigError, match="web_search"):
        build_container(s)
    allowed = write_config(LOCAL_CONFIG, strict_offline_exceptions=["web_search"])
    build_container(load_settings(allowed, {"APP_ROOT_DIR": str(tmp_path), "TOOLS__WEB_SEARCH__ENABLED": "true"}))


def test_strict_offline_cannot_be_switched_off_from_env(load_local):
    assert load_local(STRICT_OFFLINE="false").strict_offline is True


def test_cloud_config_under_strict_offline_lists_every_remote_part(write_config, tmp_path):
    strict_cloud = write_config(CLOUD_CONFIG, strict_offline=True)
    s = load_settings(strict_cloud, {"APP_ROOT_DIR": str(tmp_path), **CLOUD_SECRETS})
    problems = offline_violations(s, {"llm": "openai_compatible"})
    text = "\n".join(problems)
    for expected in (
        "llm: provider",
        "vector_store.url",
        "metadata_db.url",
        "job_queue.url",
        "ice_servers",
        "otlp_endpoint",
    ):
        assert expected in text
    assert "server.public_base_url" not in text  # inbound address, not a connection


def test_runtime_env_points_libraries_at_local_models(load_local, tmp_path):
    env: dict[str, str] = {}
    apply_runtime_env(load_local(), env)
    assert env["HF_HOME"] == str((tmp_path / "data/models/huggingface").resolve())
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
