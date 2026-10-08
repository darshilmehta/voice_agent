"""Embeddings, reranker and vector store providers (docs/DESIGN.md §3.2, §6). Methods arrive in phases 1 and 2."""

from __future__ import annotations

from .base import HealthStatus, Provider, ProviderHealth
from .models import LocalModelProvider
from .registry import register


class Embedder(Provider):
    capability = "embeddings"


@register
class BgeM3Embedder(Embedder, LocalModelProvider):
    name = "bge_m3"


class Reranker(Provider):
    capability = "reranker"


@register
class BgeReranker(Reranker, LocalModelProvider):
    name = "bge_reranker"


class VectorStore(Provider):
    capability = "vector_store"


@register
class QdrantStore(VectorStore):
    name = "qdrant"

    async def health(self) -> ProviderHealth:
        base = self.config.url.rstrip("/")  # type: ignore[attr-defined]
        key = self.config.api_key  # type: ignore[attr-defined]
        headers = {"api-key": key} if key else {}
        r, ms, err = await self._probe(f"{base}/readyz", headers=headers)
        if r is None:
            return self._health(HealthStatus.DOWN, f"Qdrant unreachable at {base} ({err})", ms)
        if r.status_code != 200:
            return self._health(HealthStatus.DOWN, f"Qdrant not ready: HTTP {r.status_code}", ms)
        collection = self.config.collection  # type: ignore[attr-defined]
        c, _, _ = await self._probe(f"{base}/collections/{collection}/exists", headers=headers)
        exists = c is not None and c.status_code == 200 and c.json().get("result", {}).get("exists", False)
        state = "exists" if exists else "will be created on first ingest"
        return self._health(HealthStatus.OK, f"ready; collection {collection!r} {state}", ms)
