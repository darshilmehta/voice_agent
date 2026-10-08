"""Auth, job queue, session store and event bus providers (docs/DESIGN.md §6.4)."""

from __future__ import annotations

from .base import HealthStatus, PlaceholderProvider, Provider, ProviderHealth
from .registry import register


class AuthProvider(Provider):
    capability = "auth"


@register
class NoAuth(AuthProvider):
    name = "none"

    async def health(self) -> ProviderHealth:
        return self._health(HealthStatus.OK, "authentication disabled (local profile)")


@register
class OIDCAuth(AuthProvider, PlaceholderProvider):
    name = "oidc"


class JobQueue(Provider):
    capability = "job_queue"


@register
class InProcessJobQueue(JobQueue):
    name = "in_process"


@register
class RedisJobQueue(JobQueue, PlaceholderProvider):
    name = "redis"


class SessionStore(Provider):
    capability = "session_store"


@register
class InMemorySessionStore(SessionStore):
    name = "in_memory"


@register
class RedisSessionStore(SessionStore, PlaceholderProvider):
    name = "redis"


class EventBus(Provider):
    capability = "event_bus"


@register
class InProcessEventBus(EventBus):
    name = "in_process"


@register
class RedisEventBus(EventBus, PlaceholderProvider):
    name = "redis"
