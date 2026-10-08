"""Live-data web search providers (docs/DESIGN.md §3.7). Search arrives in phase 8."""

from __future__ import annotations

from .base import HealthStatus, PlaceholderProvider, Provider, ProviderHealth
from .registry import register


class WebSearch(Provider):
    capability = "web_search"


@register
class SearXNGSearch(WebSearch):
    """Self-hosted SearXNG. The container is local, but it forwards queries to public engines, so it is remote."""

    name = "searxng"
    is_remote = True

    async def health(self) -> ProviderHealth:
        if not self.config.enabled:  # type: ignore[attr-defined]
            return self._health(HealthStatus.DISABLED, "off (enabled in phase 8)")
        base = (self.config.url or "").rstrip("/")  # type: ignore[attr-defined]
        r, ms, err = await self._probe(f"{base}/healthz")
        if r is None or r.status_code != 200:
            reason = err or f"HTTP {r.status_code}"  # type: ignore[union-attr]
            return self._health(HealthStatus.DOWN, f"SearXNG unreachable at {base} ({reason})", ms)
        return self._health(HealthStatus.OK, "ready", ms)


@register
class SearchAPI(WebSearch, PlaceholderProvider):
    name = "search_api"
