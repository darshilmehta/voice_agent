# Backend

FastAPI service: configuration, provider registry, health and (in later phases) ingestion, retrieval and the voice loop. Architecture: [`../docs/DESIGN.md`](../docs/DESIGN.md).

## Run

```bash
uv sync
```

```bash
uv run python -m app
```

Host and port come from the config (`127.0.0.1:8000` locally). For auto-reload during development:

```bash
uv run uvicorn --factory app.main:create_app --reload
```

The config file is `../config/local.config.json` unless `APP_CONFIG_FILE` says otherwise ([`../config/README.md`](../config/README.md)). Configuration problems (missing `${VAR}` secrets, unknown keys, placeholder providers, anything that would reach the network under `strict_offline`) stop startup with exit code 2 and a message listing every problem.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Overall status plus every provider: `ok`, `degraded`, `down`, `disabled` or `not_implemented`, with a reason |
| `GET /api/config/public` | What the browser needs (title, languages, feature flags, limits, non-secret auth settings). Never secrets |

## Test and lint

```bash
uv run pytest
```

```bash
uv run ruff check . && uv run ruff format --check .
```

Tests never touch the real Ollama, Qdrant or model files: HTTP is mocked and the project root is moved to a temp directory.

## Layout

```text
app/
  __main__.py        python -m app: validate config, then run uvicorn
  main.py            create_app(): lifespan, CORS, routers
  settings.py        JSON config schema + loader (${VAR} secrets, SECTION__KEY env overrides)
  offline.py         strict_offline guard, HF offline env
  logging_setup.py   console or JSON logs
  api/               /health, /api/config/public
  providers/
    base.py          Provider, PlaceholderProvider, health model, local_snapshot()
    registry.py      capability → provider resolution, Container
    llm.py retrieval.py storage.py ingestion.py speech.py runtime.py web_search.py
tests/
```

Each capability has a base class (for example `LLMClient`, `VectorStore`); its methods are added in the phase that builds it. A capability module becomes a package once it grows.
