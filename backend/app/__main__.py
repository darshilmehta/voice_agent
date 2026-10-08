"""Run the backend with host and port from the config: ``uv run python -m app``."""

import sys

import uvicorn

from .providers.registry import resolve_providers
from .settings import ConfigError, load_settings


def main() -> int:
    try:
        settings = load_settings()
        resolve_providers(settings)  # fail with a readable message, not a traceback from inside uvicorn
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=settings.server.host,
        port=settings.server.port,
        proxy_headers=bool(settings.server.trusted_proxy_ips),
        forwarded_allow_ips=",".join(settings.server.trusted_proxy_ips) or None,
        log_config=None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
