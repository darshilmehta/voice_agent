"""Provider base classes.

Every external capability (LLM, vector store, STT, …) is a provider chosen by name in the config.
Business logic depends on the capability's interface, never on a concrete provider (docs/DESIGN.md §6).

Capability methods are added to each capability's base class in the phase that builds them; phase 0
provides construction, lifecycle and health checks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel

from ..settings import Settings


class HealthStatus(StrEnum):
    OK = "ok"  # reachable / present and usable
    DEGRADED = "degraded"  # works partially (e.g. a configured model is missing)
    DOWN = "down"  # dependency unreachable or assets missing
    DISABLED = "disabled"  # turned off in config
    NOT_IMPLEMENTED = "not_implemented"  # placeholder provider (cloud), see PlaceholderProvider


class ProviderHealth(BaseModel):
    capability: str
    provider: str
    status: HealthStatus
    detail: str = ""
    remote: bool
    implemented: bool
    latency_ms: float | None = None


@dataclass(frozen=True)
class ProviderContext:
    """Shared resources handed to every provider."""

    settings: Settings
    http: httpx.AsyncClient

    @property
    def hf_home(self) -> Path:
        return self.settings.path(self.settings.model_cache.hf_home)


class Provider:
    capability: ClassVar[str]
    name: ClassVar[str]
    is_remote: ClassVar[bool] = False
    implemented: ClassVar[bool] = True

    def __init__(self, config: BaseModel, ctx: ProviderContext) -> None:
        self.config = config
        self.ctx = ctx

    async def start(self) -> None:
        """Prepare local resources (directories, connections). Called once at app startup."""

    async def close(self) -> None:
        """Release resources. Called once at shutdown."""

    async def health(self) -> ProviderHealth:
        return self._health(HealthStatus.OK)

    def _health(self, status: HealthStatus, detail: str = "", latency_ms: float | None = None) -> ProviderHealth:
        return ProviderHealth(
            capability=self.capability,
            provider=self.name,
            status=status,
            detail=detail,
            remote=self.is_remote,
            implemented=self.implemented,
            latency_ms=None if latency_ms is None else round(latency_ms, 1),
        )

    async def _probe(self, url: str, **kwargs: Any) -> tuple[httpx.Response | None, float, str]:
        """GET a URL; returns (response or None, latency ms, error text)."""
        t0 = time.perf_counter()
        try:
            r = await self.ctx.http.get(url, **kwargs)
            return r, (time.perf_counter() - t0) * 1000, ""
        except httpx.HTTPError as e:
            return None, (time.perf_counter() - t0) * 1000, f"{type(e).__name__}: {e}" if str(e) else type(e).__name__


class _Unimplemented:
    """Stands in for a capability method on a placeholder provider: reading the attribute raises."""

    def __init__(self, attr: str) -> None:
        self.attr = attr

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        raise NotImplementedError(
            f"{obj.capability} provider {obj.name!r} is a placeholder: {self.attr!r} is not implemented"
        )


class PlaceholderProvider(Provider):
    """A provider that exists so configuration and wiring are real, but has no implementation yet.

    Construction and health checks work; using any capability method raises NotImplementedError, including the
    methods its capability interface defines (e.g. ``ObjectStore.put`` on the ``s3`` placeholder).
    """

    is_remote = True
    implemented = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        lifecycle = set(dir(Provider))  # start, close, health, … keep working
        for klass in cls.__mro__[1:]:
            if klass in (PlaceholderProvider, Provider) or not issubclass(klass, Provider):
                continue
            for attr, value in vars(klass).items():
                if attr.startswith("_") or attr in lifecycle or attr in vars(cls) or not callable(value):
                    continue
                setattr(cls, attr, _Unimplemented(attr))

    async def health(self) -> ProviderHealth:
        return self._health(
            HealthStatus.NOT_IMPLEMENTED, f"{self.name} is a placeholder; implement it before deploying"
        )

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("__"):
            raise AttributeError(attr)
        raise NotImplementedError(
            f"{self.capability} provider {self.name!r} is a placeholder: {attr!r} is not implemented"
        )


def local_snapshot(hf_home: Path, repo_id: str) -> Path | None:
    """Newest downloaded snapshot of a Hugging Face repo, or None. Providers load from this path, never the repo id,
    so libraries can't reach the Hub (docs/DESIGN.md §9.2)."""
    snaps = hf_home / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snaps.is_dir():
        return None
    dirs = sorted((d for d in snaps.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
    return dirs[-1] if dirs else None
