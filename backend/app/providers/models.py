"""Shared logic for providers backed by locally downloaded Hugging Face models.

Heavy ML libraries (torch, FlagEmbedding, sentence-transformers, docling) live in the optional ``ml`` dependency
group and are imported lazily inside methods, so the app, its health checks and CI run without them.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import platform
import sys
from pathlib import Path
from typing import ClassVar

from .base import HealthStatus, Provider, ProviderHealth, local_snapshot

APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"

ML_INSTALL_HINT = "install the ML dependency group: cd backend && uv sync --group ml"


class ModelUnavailableError(RuntimeError):
    """A local model, or the libraries needed to run it, are not installed."""


def device_problem(device: str) -> str | None:
    """Why the configured device can't work on this machine, or None. CUDA is verified when the model loads."""
    if device in ("mps", "gpu") and not APPLE_SILICON:
        return f"device {device!r} needs Apple Silicon (this machine: {sys.platform}/{platform.machine()})"
    return None


def missing_modules(names: tuple[str, ...]) -> list[str]:
    """Top-level modules from ``names`` that can't be imported (checked without importing them)."""
    return [n for n in names if importlib.util.find_spec(n) is None]


def require_modules(owner: str, names: tuple[str, ...]) -> None:
    missing = missing_modules(names)
    if missing:
        raise ModelUnavailableError(f"{owner} needs {', '.join(missing)}: {ML_INSTALL_HINT}")


def quiet_ml_env() -> None:
    """Defaults that keep model libraries quiet and fork-safe; never overrides what the user set."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def free_torch_memory() -> None:
    """Return cached accelerator memory to the system after a model was dropped."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # best effort: freeing memory must never fail a shutdown
        pass


class LocalModelProvider(Provider):
    """Health = model files present + libraries installed + device available. Models load lazily on first use."""

    # Top-level modules the provider imports when it loads its model (the optional ``ml`` group).
    required_modules: ClassVar[tuple[str, ...]] = ()

    def model_repo(self) -> str:
        return self.config.model  # type: ignore[attr-defined]

    def extra_problems(self) -> list[str]:
        return []

    def snapshot_path(self) -> Path:
        """Local snapshot directory of the model. Models load from this path, never the repo id (§9.2)."""
        repo = self.model_repo()
        snap = local_snapshot(self.ctx.hf_home, repo)
        if snap is None:
            raise ModelUnavailableError(
                f"{repo} is not downloaded under {self.ctx.hf_home} (run scripts/setup/download_models.sh)"
            )
        return snap

    async def health(self) -> ProviderHealth:
        repo = self.model_repo()
        snap = local_snapshot(self.ctx.hf_home, repo)
        if snap is None:
            return self._health(HealthStatus.DOWN, f"{repo} not downloaded (run scripts/setup/download_models.sh)")
        device = self.config.device  # type: ignore[attr-defined]
        missing = missing_modules(self.required_modules)
        libs = [f"{', '.join(missing)} not installed ({ML_INSTALL_HINT})"] if missing else []
        problems = [p for p in (device_problem(device), *libs, *self.extra_problems()) if p]
        if problems:
            return self._health(HealthStatus.DEGRADED, "; ".join(problems))
        note = " (CUDA verified at load)" if device == "cuda" else ""
        return self._health(HealthStatus.OK, f"{repo} present, device {device}{note}; loads on first use")
