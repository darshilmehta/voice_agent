"""Shared logic for providers backed by locally downloaded Hugging Face models.

Heavy ML libraries (torch, FlagEmbedding, sentence-transformers, docling, mlx-whisper, kokoro …) live in the optional
``ml`` dependency group and are imported lazily inside methods, so the app, its health checks and CI run without them.

Models load on first use, or earlier through ``preload()`` (the app preloads the conversation models in the background
at startup, so the first question isn't slowed by loading). Inference runs off the event loop.
"""

from __future__ import annotations

import asyncio
import gc
import importlib.util
import os
import platform
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from .base import HealthStatus, Provider, ProviderContext, ProviderHealth, local_snapshot

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


class LazyModelProvider(LocalModelProvider):
    """A model loaded on first use (or by ``preload``), used by one thread at a time, and freed on close.

    Subclasses implement ``_load(snapshot)`` and optionally ``_warm(model)`` (a tiny inference run once after a
    preload, so the first real request doesn't pay for kernel compilation or lazy initialisation). Blocking work goes
    through ``_run``, which uses a worker thread (``asyncio.to_thread``); providers whose library is bound to one
    thread (MLX) override it.
    """

    def __init__(self, config: BaseModel, ctx: ProviderContext) -> None:
        super().__init__(config, ctx)
        self._lock = threading.Lock()
        self._model: Any = None
        self._load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self, snapshot: Path) -> Any:  # pragma: no cover - needs the ml group and model files
        raise NotImplementedError

    def _warm(self, model: Any) -> None:
        """Run a tiny inference right after a preload. Default: nothing."""

    def _model_locked(self) -> Any:
        """The model, loading it from the local snapshot if needed. Call with ``self._lock`` held."""
        if self._model is None:
            require_modules(f"{self.capability} provider {self.name!r}", self.required_modules)
            snapshot = self.snapshot_path()
            quiet_ml_env()
            try:
                self._model = self._load(snapshot)
            except Exception as e:
                self._load_error = f"{type(e).__name__}: {e}"
                raise
            self._load_error = None
        return self._model

    async def _run[T](self, fn: Callable[..., T], *args: Any) -> T:
        """Run blocking model work off the event loop."""
        return await asyncio.to_thread(fn, *args)

    async def preload(self) -> None:
        await self._run(self._preload_sync)

    def _preload_sync(self) -> None:
        with self._lock:
            fresh = self._model is None
            model = self._model_locked()
            if fresh:
                self._warm(model)

    def _release(self, model: Any) -> None:
        """Free library-level caches holding the model (beyond our reference). Default: nothing."""

    def unload(self) -> None:
        with self._lock:
            model, self._model = self._model, None
        if model is not None:
            self._release(model)
            del model
            free_torch_memory()

    async def close(self) -> None:
        await self._run(self.unload)

    async def health(self) -> ProviderHealth:
        report = await super().health()
        if self._model is not None and report.status == HealthStatus.OK:
            device = self.config.device  # type: ignore[attr-defined]
            return self._health(HealthStatus.OK, f"{self.model_repo()} loaded, device {device}")
        if self._load_error and report.status == HealthStatus.OK:
            return self._health(HealthStatus.DOWN, f"{self.model_repo()} failed to load: {self._load_error}")
        return report
