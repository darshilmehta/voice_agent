"""LLM providers (interface `LLMClient`, docs/DESIGN.md §6). Generation methods arrive in phase 1."""

from __future__ import annotations

from .base import HealthStatus, PlaceholderProvider, Provider, ProviderHealth
from .registry import register


class LLMClient(Provider):
    capability = "llm"


@register
class OllamaLLM(LLMClient):
    name = "ollama"

    async def health(self) -> ProviderHealth:
        base = self.config.base_url.rstrip("/")  # type: ignore[attr-defined]
        r, ms, err = await self._probe(f"{base}/api/tags")
        if r is None:
            return self._health(HealthStatus.DOWN, f"Ollama unreachable at {base} ({err})", ms)
        if r.status_code != 200:
            return self._health(HealthStatus.DOWN, f"Ollama returned HTTP {r.status_code}", ms)
        pulled = {m.get("name") for m in r.json().get("models", [])}
        wanted = dict.fromkeys([self.config.chat_model, self.config.router_model])  # type: ignore[attr-defined]
        missing = [m for m in wanted if m not in pulled and f"{m}:latest" not in pulled]
        if missing:
            return self._health(HealthStatus.DEGRADED, f"not pulled: {', '.join(missing)} (ollama pull <model>)", ms)
        return self._health(HealthStatus.OK, f"models ready: {', '.join(wanted)}", ms)


@register
class OpenAICompatibleLLM(LLMClient, PlaceholderProvider):
    name = "openai_compatible"
