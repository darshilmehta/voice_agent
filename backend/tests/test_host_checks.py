"""Read-only host checks (docs/DESIGN.md §8): Ollama's prompt cache uncapped → a warning in /health and the log."""

from __future__ import annotations

import plistlib
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.providers.registry import build_container
from app.services import host_checks
from app.services.host_checks import CACHE_RAM_FIX, HostChecks, HostWarning, prompt_cache_warnings

from .conftest import mock_http


def _agent(tmp_path: Path, env: dict[str, str]) -> Path:
    path = tmp_path / "sh.brew.ollama.plist"
    path.write_bytes(plistlib.dumps({"Label": "sh.brew.ollama", "EnvironmentVariables": env}))
    return path


def test_uncapped_on_macos_warns_with_the_command(load_local, tmp_path):
    s = load_local()
    agent = _agent(tmp_path, {"OLLAMA_FLASH_ATTENTION": "1"})
    warnings = prompt_cache_warnings(s, platform="darwin", getenv=lambda _: "", agents=lambda: [agent])
    assert [w.code for w in warnings] == ["ollama_prompt_cache_uncapped"]
    assert warnings[0].fix == "launchctl setenv LLAMA_ARG_CACHE_RAM 1024 && brew services restart ollama"
    assert "LLAMA_ARG_CACHE_RAM" in warnings[0].message


def test_capped_by_launchctl_or_by_the_launch_agent(load_local, tmp_path):
    s = load_local()
    assert prompt_cache_warnings(s, platform="darwin", getenv=lambda _: "1024", agents=lambda: []) == []
    agent = _agent(tmp_path, {"LLAMA_ARG_CACHE_RAM": "1024"})
    assert prompt_cache_warnings(s, platform="darwin", getenv=lambda _: "", agents=lambda: [agent]) == []


def test_skipped_where_it_doesnt_apply_or_cant_tell(load_local, tmp_path):
    called: list[str] = []

    def getenv(name: str) -> str:
        called.append(name)
        return ""

    s = load_local()
    assert prompt_cache_warnings(s, platform="linux", getenv=getenv, agents=lambda: []) == []
    remote = load_local(LLM__BASE_URL="http://ollama.internal:11434")
    assert prompt_cache_warnings(remote, platform="darwin", getenv=getenv, agents=lambda: []) == []
    assert called == []  # nothing run off macOS or for a remote Ollama
    # launchctl missing or failing, and no launch agent says: no false alarm
    assert prompt_cache_warnings(s, platform="darwin", getenv=lambda _: None, agents=lambda: []) == []
    broken = tmp_path / "homebrew.mxcl.ollama.plist"
    broken.write_bytes(b"not a plist")
    assert prompt_cache_warnings(s, platform="darwin", getenv=lambda _: None, agents=lambda: [broken]) == []


async def test_checks_are_cached_and_rerun_after_the_ttl(load_local):
    runs: list[int] = []
    warning = HostWarning(code="ollama_prompt_cache_uncapped", message="m", fix=CACHE_RAM_FIX, docs="d")

    def check(_settings) -> list[HostWarning]:
        runs.append(1)
        return [warning] if len(runs) == 1 else []

    checks = HostChecks(load_local(), ttl_s=0.0, check=check)
    assert await checks.warnings() == [warning]
    assert await checks.warnings() == []  # fixed meanwhile: the warning goes
    cached = HostChecks(load_local(), ttl_s=3600, check=check)
    await cached.warnings()
    await cached.warnings()
    assert len(runs) == 3


def test_health_reports_the_warning(load_local, monkeypatch):
    logged: list[str] = []
    monkeypatch.setattr(host_checks.log, "warning", lambda msg, *args: logged.append(msg % args))
    s = load_local()
    warning = HostWarning(code="ollama_prompt_cache_uncapped", message="uncapped", fix=CACHE_RAM_FIX, docs="d")
    checks = HostChecks(s, check=lambda _: [warning])
    app = create_app(s, build_container(s, http=mock_http()), preload_models=False, host_checks=checks)
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["warnings"] == [warning.model_dump()]
    assert any(CACHE_RAM_FIX in line for line in logged)  # logged at startup

    quiet = HostChecks(s, check=lambda _: [])
    app = create_app(s, build_container(s, http=mock_http()), preload_models=False, host_checks=quiet)
    with TestClient(app) as client:
        assert client.get("/health").json()["warnings"] == []
