"""Strict-offline guard (docs/DESIGN.md §6.3).

With ``strict_offline: true`` the app must not reach anything outside this machine. Startup fails if
any configured provider is remote or any configured URL is not loopback, unless that capability is
listed in ``strict_offline_exceptions`` (currently only ``web_search`` may be). Hostnames listed in
``strict_offline_local_hosts`` (Docker Compose services, ``host.docker.internal``) count as this machine.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Collection, Iterator, Mapping
from typing import Any
from urllib.parse import urlparse

from .settings import Settings

# URL-valued settings that describe *inbound* addresses, not places the backend connects to.
_INBOUND_URL_FIELDS = {"server.public_base_url"}
_URL_KEYS = ("url", "urls", "base_url", "endpoint_url", "issuer_url", "jwks_url", "otlp_endpoint")


def is_loopback_url(url: str, local_hosts: Collection[str] = ()) -> bool:
    """True for URLs that can only reach this machine: loopback hosts, local files, sqlite paths, and any
    hostname explicitly listed as local (``strict_offline_local_hosts``)."""
    parsed = urlparse(url)
    if parsed.scheme in ("file", "") or parsed.scheme.startswith("sqlite"):
        return True
    host = parsed.hostname
    if host is None:
        # stun:/turn: style URIs have no "//"; their host is in the path
        host = parsed.path.split(":")[0] or None
    if host is None:
        return False
    if host == "localhost" or host.endswith(".localhost") or host in local_hosts:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _url_fields(node: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(node, Mapping):
        for key, value in node.items():
            child = f"{path}.{key}" if path else key
            if key in _URL_KEYS:
                if isinstance(value, str):
                    yield child, value
                elif isinstance(value, list):
                    yield from ((child, v) for v in value if isinstance(v, str))
            else:
                yield from _url_fields(value, child)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            yield from _url_fields(item, f"{path}[{i}]")


def offline_violations(settings: Settings, remote_capabilities: Mapping[str, str]) -> list[str]:
    """Everything that would reach the network. ``remote_capabilities`` maps capability → remote provider name."""
    excepted = set(settings.strict_offline_exceptions)
    local_hosts = set(settings.strict_offline_local_hosts)
    web_search = settings.tools.web_search
    problems = [
        f"{cap}: provider {name!r} is remote" for cap, name in remote_capabilities.items() if cap not in excepted
    ]
    data = settings.model_dump(mode="json")
    for path, url in _url_fields(data):
        if path in _INBOUND_URL_FIELDS or is_loopback_url(url, local_hosts):
            continue
        if path.startswith("tools.web_search.") and (not web_search.enabled or "web_search" in excepted):
            continue
        problems.append(f"{path}: {url} is not loopback or a listed local host")
    if web_search.enabled and "web_search" not in excepted:
        problems.append('tools.web_search is enabled but "web_search" is not in strict_offline_exceptions')
    return problems


def apply_runtime_env(settings: Settings, environ: dict[str, str] | None = None) -> None:
    """Point model libraries at the project's model cache and, when strict, forbid them from going online."""
    environ = os.environ if environ is None else environ
    environ["HF_HOME"] = str(settings.path(settings.model_cache.hf_home))
    environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if settings.strict_offline:
        environ["HF_HUB_OFFLINE"] = "1"
        environ["TRANSFORMERS_OFFLINE"] = "1"
