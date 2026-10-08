# Backend

FastAPI service: configuration, provider registry, health, projects/chats/transcripts persistence and (in later phases) ingestion, retrieval and the voice loop. Architecture: [`../docs/DESIGN.md`](../docs/DESIGN.md).

## Run

```bash
uv sync
```

Ingestion and retrieval run local models (Docling, BGE-M3, bge-reranker-v2-m3) from the optional `ml` dependency group. Without it the app starts and `/health` marks those providers `degraded`:

```bash
uv sync --group ml
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
| `GET` / `POST /api/projects` | List projects (pinned first, then most recent activity; `?include_archived=true`) / create one |
| `GET` / `PATCH` / `DELETE /api/projects/{id}` | One project / rename, describe, `pinned`, `archived` / delete with everything inside it |
| `GET /api/projects/{id}/documents` | The project's documents, newest first (upload arrives with ingestion) |
| `GET` / `POST /api/projects/{id}/chats` | The project's chats, most recent activity first / create one |
| `GET` / `PATCH` / `DELETE /api/chats/{id}` | One chat / rename, `pinned`, `archived`, `document_scope` (null = all documents), `language` / delete with its messages |
| `GET /api/chats/{id}/messages` | The transcript, paginated: `?after=<seq>` reads forward, `?before=<seq>` reads backward, `limit` ≤ 200 |
| `GET /api/pins` | Pinned projects and chats for the sidebar, most recently pinned first |

Lists return `{"items": [...]}`. Unknown ids are `404`, invalid input `422`, both with a `detail` message. `PATCH` changes only the fields sent; unknown fields are rejected. A transcript page is `{"items", "total", "has_more", "next_cursor"}`, items always in chronological order; pass `next_cursor` back as the same parameter (`after` or `before`) for the next page. To open a chat at its latest messages, request `before=<message_count + 1>`.

## Database

Projects, documents, chats, messages and summaries live in SQLite (`data/sqlite/app.db`) through SQLAlchemy, so Postgres is a URL change plus implementing the `postgres` provider (docs/DESIGN.md §3.9, §6). Ids are readable and sortable: `prj_…`, `doc_…`, `cht_…`, `msg_…` (ULID layout). Deleting a project or chat cascades in the database (`ON DELETE CASCADE`, foreign keys enforced on every SQLite connection).

The schema is managed by Alembic. The app upgrades to the latest revision at startup, so there is nothing to run by hand. After changing `app/db/models.py`, generate a migration (it is linted and formatted automatically) and review it:

```bash
uv run alembic revision --autogenerate --rev-id 0002 -m "describe the change"
```

The CLI migrates the database the app config points at (`APP_CONFIG_FILE`; `METADATA_DB__URL` overrides it): `uv run alembic upgrade head`, `uv run alembic current`, `uv run alembic check`. A test fails if the models and the migrations disagree.

## Test and lint

```bash
uv run pytest
```

```bash
uv run ruff check . && uv run ruff format --check .
```

Tests never touch the real Ollama, Qdrant, model files or database: HTTP is mocked and the project root is moved to a temp directory, so every test gets a fresh, migrated SQLite database. Tests that need the `ml` group skip without it.

Integration tests (real models, Docling and a running Qdrant; a temporary collection is created and dropped) are opt-in:

```bash
RUN_INTEGRATION=1 uv run --group ml pytest tests/integration
```

They read models from `MODELS_ROOT` (default `../data/models`) and the smoke-test PDF from `SMOKE_DOCS` (default `../data/smoke/docs`, written by `scripts/smoke/05_docling.py`); Qdrant from `QDRANT_URL` (default `http://127.0.0.1:6333`). Timings and ranks print in the summary.

## Layout

```text
alembic.ini          Alembic CLI config (the app migrates without it)
app/
  __main__.py        python -m app: validate config, then run uvicorn
  main.py            create_app(): lifespan, CORS, routers, error handlers
  settings.py        JSON config schema + loader (${VAR} secrets, SECTION__KEY env overrides)
  offline.py         strict_offline guard, HF offline env
  logging_setup.py   console or JSON logs
  api/               /health, /api/config/public, projects, chats (+ messages), pins; deps.py wires services
  db/
    models.py        SQLAlchemy models: projects, documents, document_versions, ingestion_jobs, chats, messages, chat_summaries
    types.py         UTC datetimes, JSON/JSONB, new_id()
    engine.py        metadata_db.url → engine (SQLite: foreign keys on, WAL)
    migrate.py       upgrade/downgrade from code
    migrations/      Alembic env.py + versions/
  domain/            Pydantic objects the services return (Project, Chat, Message, …)
  services/          business logic on the MetadataDB interface: projects, chats, messages, summaries, pins, documents
  providers/
    base.py          Provider, PlaceholderProvider, health model, local_snapshot()
    registry.py      capability → provider resolution, Container
    models.py        local-model health, lazy ML imports (ml group), ModelUnavailableError
    ingestion.py     DoclingParser: ParsedDocument (pages, items, tables + cells), chunking with provenance
    retrieval.py     BgeM3Embedder (dense + sparse), BgeReranker, QdrantStore (hybrid RRF search)
    llm.py storage.py speech.py runtime.py web_search.py
  services/
    ingestion.py     ingest_file: parse → chunk → embed → upsert (no DB writes)
    retrieval.py     hybrid search → rerank → top N + confidence signal
tests/
  integration/       opt-in, real models + Qdrant (RUN_INTEGRATION=1)
```

Each capability has a base class (for example `LLMClient`, `VectorStore`); its methods are added in the phase that builds it. A capability module becomes a package once it grows.
