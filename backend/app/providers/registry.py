"""Provider registry and container: config names → provider instances."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel

from ..offline import offline_violations
from ..settings import ConfigError, Settings
from .base import HealthStatus, PlaceholderProvider, Provider, ProviderContext, ProviderHealth

# Every capability and where its config lives. Order is the order /health reports them in.
CAPABILITIES: dict[str, Callable[[Settings], BaseModel]] = {
    "llm": lambda s: s.llm,
    "embeddings": lambda s: s.embeddings,
    "reranker": lambda s: s.reranker,
    "vector_store": lambda s: s.vector_store,
    "metadata_db": lambda s: s.metadata_db,
    "object_store": lambda s: s.object_store,
    "ingestion": lambda s: s.ingestion,
    "stt": lambda s: s.stt,
    "vad": lambda s: s.vad,
    "tts": lambda s: s.tts,
    "audio_transport": lambda s: s.audio_transport,
    "auth": lambda s: s.auth,
    "job_queue": lambda s: s.job_queue,
    "session_store": lambda s: s.session_store,
    "event_bus": lambda s: s.event_bus,
    "web_search": lambda s: s.tools.web_search,
}

_REGISTRY: dict[tuple[str, str], type[Provider]] = {}

HEALTH_TIMEOUT_S = 3.0


def register[P: type[Provider]](cls: P) -> P:
    """Class decorator: make a provider selectable by (capability, name)."""
    if cls.capability not in CAPABILITIES:
        raise ValueError(f"{cls.__name__}: unknown capability {cls.capability!r}")
    key = (cls.capability, cls.name)
    if key in _REGISTRY:
        raise ValueError(f"provider {key} registered twice")
    _REGISTRY[key] = cls
    return cls


def registered(capability: str) -> dict[str, type[Provider]]:
    return {name: cls for (cap, name), cls in _REGISTRY.items() if cap == capability}


def provider_class(capability: str, name: str) -> type[Provider]:
    try:
        return _REGISTRY[(capability, name)]
    except KeyError:
        known = ", ".join(sorted(registered(capability))) or "none"
        raise ConfigError(f"{capability}: unknown provider {name!r} (available: {known})") from None


@dataclass
class Container:
    """All configured providers plus the shared HTTP client."""

    settings: Settings
    http: httpx.AsyncClient
    providers: dict[str, Provider] = field(default_factory=dict)

    def __getitem__(self, capability: str) -> Provider:
        return self.providers[capability]

    @property
    def placeholders(self) -> list[str]:
        return [cap for cap, p in self.providers.items() if isinstance(p, PlaceholderProvider)]

    async def start(self) -> None:
        for p in self.providers.values():
            await p.start()

    async def close(self) -> None:
        for p in reversed(list(self.providers.values())):
            await p.close()
        await self.http.aclose()

    async def health(self) -> list[ProviderHealth]:
        async def check(p: Provider) -> ProviderHealth:
            try:
                return await asyncio.wait_for(p.health(), HEALTH_TIMEOUT_S)
            except TimeoutError:
                return p._health(HealthStatus.DOWN, f"health check timed out after {HEALTH_TIMEOUT_S:.0f}s")
            except Exception as e:  # a broken health check must not break /health
                return p._health(HealthStatus.DOWN, f"health check failed: {type(e).__name__}: {e}")

        return list(await asyncio.gather(*(check(p) for p in self.providers.values())))


def resolve_providers(settings: Settings, *, allow_placeholders: bool = False) -> dict[str, type[Provider]]:
    """Check the config against the registry and the offline guard; return the provider class per capability.

    Raises ConfigError listing every problem. ``allow_placeholders`` lets tests wire configs that name
    unimplemented (placeholder) providers; the running app refuses them so a deployment can't start
    half-implemented.
    """
    from . import load_all  # importing registers every provider module

    load_all()
    errors: list[str] = []
    classes: dict[str, type[Provider]] = {}
    for capability, section in CAPABILITIES.items():
        try:
            classes[capability] = provider_class(capability, section(settings).provider)  # type: ignore[attr-defined]
        except ConfigError as e:
            errors.append(str(e))
    if errors:
        raise ConfigError("provider configuration errors:\n  - " + "\n  - ".join(errors))

    placeholders = sorted(f"{cap}={cls.name}" for cap, cls in classes.items() if issubclass(cls, PlaceholderProvider))
    if placeholders and not allow_placeholders:
        raise ConfigError(
            "config selects placeholder providers that are not implemented yet: "
            + ", ".join(placeholders)
            + ". Implement them (docs/DESIGN.md §6.4) or choose implemented providers."
        )

    if settings.strict_offline:
        # A disabled web search never runs, so it doesn't count as reaching the network.
        inactive = set() if settings.tools.web_search.enabled else {"web_search"}
        remote = {cap: cls.name for cap, cls in classes.items() if cls.is_remote and cap not in inactive}
        problems = offline_violations(settings, remote)
        if problems:
            raise ConfigError(
                "strict_offline is on, but the config reaches the network:\n  - " + "\n  - ".join(problems)
            )
    return classes


def build_container(
    settings: Settings,
    *,
    allow_placeholders: bool = False,
    http: httpx.AsyncClient | None = None,
) -> Container:
    """Resolve and instantiate every provider (see ``resolve_providers`` for the checks)."""
    classes = resolve_providers(settings, allow_placeholders=allow_placeholders)
    http = http or httpx.AsyncClient(timeout=httpx.Timeout(HEALTH_TIMEOUT_S, connect=1.0))
    ctx = ProviderContext(settings=settings, http=http)
    container = Container(settings=settings, http=http)
    for capability, cls in classes.items():
        container.providers[capability] = cls(CAPABILITIES[capability](settings), ctx)
    return container
