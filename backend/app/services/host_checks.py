"""Read-only checks of the machine the backend runs on, reported in ``/health`` (docs/DESIGN.md §8, setup notes).

Ollama 0.40's llama-server keeps the KV state of every distinct prompt it has served in RAM, up to 8 GiB, unless its
environment sets ``LLAMA_ARG_CACHE_RAM``; on a 16 GB Mac that pushed the machine into heavy swap (12.6 of 13.3 GB in
the last real run). The app must not change the user's Ollama or system settings, so it only looks: on macOS, with a
local Ollama, it reads ``launchctl getenv LLAMA_ARG_CACHE_RAM`` (what a ``brew services`` Ollama started after
``launchctl setenv`` inherits) and the Ollama launch agent's ``EnvironmentVariables`` (the lasting setting), and when
neither sets it reports a warning with the one command that fixes it. Anywhere else (Linux, a cloud LLM, a remote
Ollama, no ``launchctl``) the check is skipped. Nothing here writes anything or blocks startup for long (one
``launchctl`` call, 2 s at most, and one small plist read).
"""

from __future__ import annotations

import asyncio
import logging
import plistlib
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel

from ..settings import Settings

log = logging.getLogger(__name__)

CACHE_RAM_VAR = "LLAMA_ARG_CACHE_RAM"
CACHE_RAM_FIX = f"launchctl setenv {CACHE_RAM_VAR} 1024 && brew services restart ollama"
LAUNCHCTL_TIMEOUT_S = 2.0
_LOOPBACK = {"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"}


class HostWarning(BaseModel):
    """Something about the machine the user should fix themselves (the app never changes it). ``fix``: the command."""

    code: Literal["ollama_prompt_cache_uncapped"]
    message: str
    fix: str
    docs: str


def _launchctl_getenv(name: str) -> str | None:
    """``launchctl getenv NAME``: the value, "" when unset, None when it can't be read."""
    launchctl = shutil.which("launchctl")
    if launchctl is None:
        return None
    try:
        done = subprocess.run(  # a fixed, read-only command
            [launchctl, "getenv", name], capture_output=True, text=True, timeout=LAUNCHCTL_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _launch_agents() -> list[Path]:
    """The Ollama launch agents of this user (``brew services`` writes ``homebrew.mxcl.ollama.plist`` or
    ``sh.brew.ollama.plist``)."""
    folder = Path.home() / "Library" / "LaunchAgents"
    try:
        return sorted(p for p in folder.glob("*ollama*.plist") if p.is_file())
    except OSError:
        return []


def _plist_value(paths: Iterable[Path], name: str) -> str | None:
    """``name`` in the ``EnvironmentVariables`` of the first plist that sets it."""
    for path in paths:
        try:
            with path.open("rb") as f:
                data = plistlib.load(f)
        except (OSError, plistlib.InvalidFileException, ValueError):
            continue
        env = data.get("EnvironmentVariables") if isinstance(data, dict) else None
        value = env.get(name) if isinstance(env, dict) else None
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def applies(settings: Settings, platform: str = sys.platform) -> bool:
    """The check means something here: macOS, the local profile's Ollama on this machine."""
    if platform != "darwin" or settings.llm.provider != "ollama":
        return False
    host = (urlparse(settings.llm.base_url).hostname or "").lower()
    return host in _LOOPBACK


def prompt_cache_warnings(
    settings: Settings,
    *,
    platform: str = sys.platform,
    getenv: Callable[[str], str | None] = _launchctl_getenv,
    agents: Callable[[], list[Path]] = _launch_agents,
) -> list[HostWarning]:
    """A warning when Ollama's prompt cache is uncapped (§8); [] when it is capped, the check doesn't apply, or it
    can't tell (``launchctl`` missing or failing: never a false alarm)."""
    if not applies(settings, platform):
        return []
    value = getenv(CACHE_RAM_VAR)
    if value:
        return []
    if _plist_value(agents(), CACHE_RAM_VAR):
        return []
    if value is None:  # launchctl couldn't be read and no launch agent sets it: unknown, say nothing
        return []
    return [
        HostWarning(
            code="ollama_prompt_cache_uncapped",
            message=(
                f"Ollama's prompt cache is uncapped ({CACHE_RAM_VAR} is not set): it can grow to 8 GB and push a "
                "16 GB Mac into heavy swap."
            ),
            fix=CACHE_RAM_FIX,
            docs="docs/DESIGN.md §8 and §9 (setup notes)",
        )
    ]


def host_warnings(settings: Settings) -> list[HostWarning]:
    """Every host check (today: the prompt cache). Never raises."""
    try:
        return prompt_cache_warnings(settings)
    except Exception as e:  # a check of the host must never stop the app
        log.info("host check skipped: %s", e)
        return []


class HostChecks:
    """The host checks for ``/health``: run at startup (logged) and again at most every ``ttl_s`` seconds, off the
    event loop, so the warning goes once the user has fixed it."""

    def __init__(
        self, settings: Settings, *, ttl_s: float = 30.0, check: Callable[[Settings], list[HostWarning]] = host_warnings
    ) -> None:
        self.settings = settings
        self.ttl_s = ttl_s
        self._check = check
        self._at: float | None = None
        self._warnings: list[HostWarning] = []

    async def warnings(self, *, log_them: bool = False) -> list[HostWarning]:
        now = time.monotonic()
        if self._at is None or now - self._at >= self.ttl_s:
            self._warnings = await asyncio.to_thread(self._check, self.settings)
            self._at = now
            if log_them:
                for w in self._warnings:
                    log.warning("%s Fix (until the next reboot): %s", w.message, w.fix)
        return list(self._warnings)
