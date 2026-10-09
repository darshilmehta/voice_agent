from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.settings import ConfigError, load_settings

from .conftest import CLOUD_CONFIG, CLOUD_SECRETS, DOCKER_CONFIG, LOCAL_CONFIG


def _keys(node: object, prefix: str = "") -> set[str]:
    if not isinstance(node, dict):
        return set()
    out: set[str] = set()
    for k, v in node.items():
        out.add(prefix + k)
        out |= _keys(v, prefix + k + ".")
    return out


def test_local_config_validates(load_local):
    s = load_local()
    assert s.profile == "local"
    assert s.strict_offline is True
    assert s.llm.chat_model == "qwen3:4b-instruct"


def test_cloud_config_validates_with_secrets(cloud_settings):
    assert cloud_settings.profile == "cloud"
    assert cloud_settings.llm.api_key == CLOUD_SECRETS["LLM_API_KEY"]
    assert CLOUD_SECRETS["DB_PASSWORD"] in cloud_settings.metadata_db.url  # interpolated inside a longer string


def test_config_files_share_every_key():
    local, cloud, docker = (json.loads(f.read_text()) for f in (LOCAL_CONFIG, CLOUD_CONFIG, DOCKER_CONFIG))
    assert _keys(local) == _keys(cloud) == _keys(docker)


def test_local_hosts_must_be_bare_hostnames(write_config, tmp_path):
    bad = write_config(LOCAL_CONFIG, strict_offline_local_hosts=["http://qdrant:6333"])
    with pytest.raises(ConfigError, match="bare hostnames"):
        load_settings(bad, {"APP_ROOT_DIR": str(tmp_path)})


def test_missing_secrets_are_all_reported(tmp_path):
    with pytest.raises(ConfigError) as err:
        load_settings(CLOUD_CONFIG, {"APP_ROOT_DIR": str(tmp_path)})
    for name in CLOUD_SECRETS:
        assert name in str(err.value)


def test_env_overrides_are_nested_and_typed(load_local):
    s = load_local(
        LLM__CHAT_MODEL="qwen3:8b",
        LLM__NUM_CTX="4096",
        TOOLS__WEB_SEARCH__ENABLED="true",
        CLIENT__LANGUAGES='["en"]',
    )
    assert s.llm.chat_model == "qwen3:8b"
    assert s.llm.num_ctx == 4096
    assert s.tools.web_search.enabled is True
    assert s.client.languages == ["en"]


def test_unrelated_env_vars_are_ignored(load_local):
    s = load_local(PATH="/usr/bin", SOMETHING__ELSE="x", HOME="/tmp")
    assert s.profile == "local"


def test_override_of_unknown_key_is_rejected(load_local):
    with pytest.raises(ConfigError, match=r"unknown key llm\.chat_modle"):
        load_local(LLM__CHAT_MODLE="typo")


def test_unknown_key_in_file_is_rejected(tmp_path):
    raw = json.loads(LOCAL_CONFIG.read_text())
    raw["llm"]["temprature"] = 0.2
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(raw))
    with pytest.raises(ConfigError, match=r"llm\.temprature"):
        load_settings(f, {"APP_ROOT_DIR": str(tmp_path)})


def test_invalid_values_are_reported(load_local):
    with pytest.raises(ConfigError, match=r"server\.port"):
        load_local(SERVER__PORT="70000")


def test_ollama_cloud_models_are_rejected(load_local):
    with pytest.raises(ConfigError, match="cloud-hosted"):
        load_local(LLM__CHAT_MODEL="gpt-oss:120b-cloud")


def test_ocr_engine_auto_is_rejected(load_local):
    with pytest.raises(ConfigError, match=r"ingestion\.ocr_engine"):
        load_local(INGESTION__OCR_ENGINE="auto")


def test_default_language_must_be_offered(load_local):
    with pytest.raises(ConfigError, match="default_language"):
        load_local(CLIENT__LANGUAGES='["hi"]', CLIENT__DEFAULT_LANGUAGE="en")


def test_relative_paths_resolve_against_project_root(load_local, tmp_path: Path):
    s = load_local()
    assert s.path(s.model_cache.hf_home) == (tmp_path / "data/models/huggingface").resolve()
    assert s.path("/abs/elsewhere") == Path("/abs/elsewhere")


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "nope.json", {})


def test_canvas_section(load_local):
    s = load_local()
    c = s.canvas
    assert (c.planner_timeout_ms, c.planner_max_tokens, c.planner_candidates, c.max_panels, c.overview_panels) == (
        8000,
        200,
        4,
        12,
        3,
    )
    assert load_local(CANVAS__OVERVIEW_PANELS="0").canvas.overview_panels == 0  # no overview
    with pytest.raises(ConfigError, match=r"canvas\.planner_candidates"):
        load_local(CANVAS__PLANNER_CANDIDATES="20")
    with pytest.raises(ConfigError, match=r"canvas\.max_panels"):
        load_local(CANVAS__MAX_PANELS="0")
    with pytest.raises(ConfigError, match=r"canvas\.planner_timeout_ms"):
        load_local(CANVAS__PLANNER_TIMEOUT_MS="-1")
