"""Shared health logic for providers backed by locally downloaded Hugging Face models."""

from __future__ import annotations

import platform
import sys

from .base import HealthStatus, Provider, ProviderHealth, local_snapshot

APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"


def device_problem(device: str) -> str | None:
    """Why the configured device can't work on this machine, or None. CUDA is verified when the model loads."""
    if device in ("mps", "gpu") and not APPLE_SILICON:
        return f"device {device!r} needs Apple Silicon (this machine: {sys.platform}/{platform.machine()})"
    return None


class LocalModelProvider(Provider):
    """Health = model files present + device available. Models load lazily on first use, not at startup."""

    def model_repo(self) -> str:
        return self.config.model  # type: ignore[attr-defined]

    def extra_problems(self) -> list[str]:
        return []

    async def health(self) -> ProviderHealth:
        repo = self.model_repo()
        snap = local_snapshot(self.ctx.hf_home, repo)
        if snap is None:
            return self._health(HealthStatus.DOWN, f"{repo} not downloaded (run scripts/setup/download_models.sh)")
        device = self.config.device  # type: ignore[attr-defined]
        problems = [p for p in (device_problem(device), *self.extra_problems()) if p]
        if problems:
            return self._health(HealthStatus.DEGRADED, "; ".join(problems))
        note = " (CUDA verified at load)" if device == "cuda" else ""
        return self._health(HealthStatus.OK, f"{repo} present, device {device}{note}; loads on first use")
