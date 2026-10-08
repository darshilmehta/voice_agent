# Backend

FastAPI service: configuration, provider registry, health, projects/chats/transcripts persistence, document upload and ingestion, text chat answered from the documents with citations, and (in later phases) the voice loop. Architecture: [`../docs/DESIGN.md`](../docs/DESIGN.md).

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
| `GET` / `POST /api/projects/{id}/documents` | The project's documents, newest first / upload one (below) |
| `GET` / `DELETE /api/documents/{id}` | One document (poll it for its status) / delete it with its vectors and stored files |
| `GET` / `POST /api/projects/{id}/chats` | The project's chats, most recent activity first / create one |
| `GET` / `PATCH` / `DELETE /api/chats/{id}` | One chat / rename, `pinned`, `archived`, `document_scope` (null = all documents), `language` / delete with its messages |
| `GET /api/chats/{id}/messages` | The transcript, paginated: `?after=<seq>` reads forward, `?before=<seq>` reads backward, `limit` ≤ 200 |
| `POST /api/chats/{id}/messages` | Ask in text: `{"text", "language": "en" \| "hi" \| null}` → the answer as Server-Sent Events (below) |
| `GET /api/pins` | Pinned projects and chats for the sidebar, most recently pinned first |

Lists return `{"items": [...]}`. Unknown ids are `404`, invalid input `422`, both with a `detail` message. `PATCH` changes only the fields sent; unknown fields are rejected. A transcript page is `{"items", "total", "has_more", "next_cursor"}`, items always in chronological order; pass `next_cursor` back as the same parameter (`after` or `before`) for the next page. To open a chat at its latest messages, request `before=<message_count + 1>`. A `503` means a dependency the operation needs is down (e.g. Qdrant during a delete); nothing was changed.

### Documents

Upload is `multipart/form-data` with one field, `file`. The extension must be in `ingestion.allowed_extensions` and the size within `server.max_upload_mb`; the content is sniffed (PDF signature, DOCX/PPTX package parts, text without binary data), the declared content type is ignored. Rejections are `422`. Identical content already in the project returns `200` with that document; a new file returns `202` with a new document in status `PENDING` (re-uploading a `FAILED` document's content retries it: `202`, same id). The original goes to the object store under a generated key (`<project>/<document>/v1.pdf`, never the uploaded name) and ingestion runs in the background job queue: `PENDING` → `PROCESSING` → `READY` (with `page_count`, `chunk_count`) or `FAILED` (with `error`). Each attempt is an `ingestion_jobs` row; parsed tables are stored cell by cell in `document_tables` (datasets for the visual canvas, docs/DESIGN.md §12.1). Ingestion models are released when the queue drains, and ingestions interrupted by a restart are queued again at startup. Deleting a document removes its vectors first (a `503` if Qdrant is down, nothing deleted), then its records, then its files; chats scoped to it drop it from `document_scope` (a scope left empty becomes null, all documents). Deleting a project does the same for everything in it.

### Chat

`POST /api/chats/{id}/messages` answers with `text/event-stream`. Each event is `event: <name>`, one `data: <JSON>` line and a blank line, in this order:

| Event | Data |
|---|---|
| `user_message` | the saved user message (same shape as the transcript's items) |
| `sources` | `{"sources": [Citation], "confidence": {"top_score", "gap", "dense_similarity", "above_threshold"}, "abstained": bool}` |
| `delta` | `{"text": "…"}`, zero or more: the answer as it is generated |
| `agent_message` | the saved agent message: final text and the citations it actually makes |
| `error` | `{"detail", "stage": "retrieval" \| "llm" \| "storage"}`; the stream ends (the user message stays saved) |

A `Citation` is `{"source_id": "S1", "document_id", "filename", "page_start", "page_end", "chunk_id", "snippet"}`; answers mark sources as `[S1]`. Unknown chat or invalid text fail before the stream (`404` / `422`).

Every turn (phase 1) is a document question: retrieval runs within the chat's project, narrowed to its `document_scope`, over `READY` documents only. If the best passage scores below `retrieval.min_rerank_score` the agent abstains with a fixed sentence in the user's language and no model call (`abstained: true`, no sources). Otherwise the top passages become numbered sources (deduplicated, grouped by section, within `retrieval.context_token_budget`) and `qwen3:4b-instruct` answers from them only, citing `[S#]`; markers naming no source are removed from the saved text, and lists are normalized to `[S1][S2]`. The answer language is `language` when given, else Hindi when at least as many words are in Devanagari as in Latin script, else English. The last 6 messages go along as context. The agent message records the route (top-level `abstained` and `stopped` booleans, confidence, model, prompt version) and latency (retrieval, rerank, first delta, total).

If the client disconnects mid-answer, generation is cancelled (the stream to Ollama is closed) and the text generated so far is saved as the agent message with `route.stopped: true`, citing only the sources that text cites; if no text was generated yet, no agent message is saved.

The turn itself is `ChatTurnService` (`app/services/chat_turns.py`), which knows nothing about HTTP: `begin(chat_id, text, language=…, modality="text" | "voice", length="short" | "full")` validates, and `run(turn)` yields typed events (`UserMessageEvent`, `SourcesEvent`, `DeltaEvent`, `AgentMessageEvent`, `ErrorEvent`, each with `name` and `payload()`). Cancelling the consumer or `aclose()` on the events stops the answer as above. This endpoint only serialises the events; the voice WebSocket will run the same turns with `modality="voice"` and speak the deltas. `length="short"` (the default, and what this endpoint uses for now) asks for 1–3 speakable sentences with a 384-token cap; `"full"` asks for a fuller written answer (768).

## Database

Projects, documents, chats, messages and summaries live in SQLite (`data/sqlite/app.db`) through SQLAlchemy, so Postgres is a URL change plus implementing the `postgres` provider (docs/DESIGN.md §3.9, §6). Ids are readable and sortable: `prj_…`, `doc_…`, `cht_…`, `msg_…` (ULID layout). Deleting a project or chat cascades in the database (`ON DELETE CASCADE`, foreign keys enforced on every SQLite connection).

The schema is managed by Alembic. The app upgrades to the latest revision at startup, so there is nothing to run by hand. After changing `app/db/models.py`, generate a migration (it is linted and formatted automatically) and review it:

```bash
uv run alembic revision --autogenerate --rev-id 0003 -m "describe the change"
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
RUN_INTEGRATION=1 uv run --group ml pytest tests/integration -s
```

They read models from `MODELS_ROOT` (default `../data/models`) and the smoke-test documents from `SMOKE_DOCS` (default `../data/smoke/docs`, written by `scripts/smoke/05_docling.py`); Qdrant from `QDRANT_URL` (default `http://127.0.0.1:6333`). `test_chat_e2e.py` also needs Ollama with `qwen3:4b-instruct` and checks phase 1 end to end over HTTP: upload until `READY`, English and Hindi fact questions cited to page 2, an out-of-document question abstaining, two documents told apart, nothing leaking across projects. Timings, ranks and answers print in the summary.

## Layout

```text
alembic.ini          Alembic CLI config (the app migrates without it)
app/
  __main__.py        python -m app: validate config, then run uvicorn
  main.py            create_app(): lifespan, CORS, routers, error handlers
  settings.py        JSON config schema + loader (${VAR} secrets, SECTION__KEY env overrides)
  offline.py         strict_offline guard, HF offline env
  logging_setup.py   console or JSON logs
  api/               /health, /api/config/public, projects, documents (upload, delete), chats (+ messages, SSE), pins;
                     deps.py wires services
  db/
    models.py        SQLAlchemy models: projects, documents, document_versions, ingestion_jobs, document_tables, chats,
                     messages, chat_summaries
    types.py         UTC datetimes, JSON/JSONB, new_id()
    engine.py        metadata_db.url → engine (SQLite: foreign keys on, WAL)
    migrate.py       upgrade/downgrade from code
    migrations/      Alembic env.py + versions/
  domain/            Pydantic objects the services return (Project, Chat, Message, Citation, …)
  services/          business logic on the provider interfaces: projects, chats, messages, summaries, pins, documents
  providers/
    base.py          Provider, PlaceholderProvider, health model, local_snapshot()
    registry.py      capability → provider resolution, Container
    models.py        local-model health, lazy ML imports (ml group), ModelUnavailableError
    ingestion.py     DoclingParser: ParsedDocument (pages, items, tables + cells), chunking with provenance
    retrieval.py     BgeM3Embedder (dense + sparse), BgeReranker, QdrantStore (hybrid RRF search)
    llm.py           OllamaLLM: streamed chat + JSON-schema output (think off, num_ctx, keep_alive, temperatures)
    storage.py       SqliteDB (metadata), FilesystemStore (object store: safe generated keys, temp copies)
    runtime.py       InProcessJobQueue (asyncio workers, idle hook, graceful shutdown), auth, sessions, events
    speech.py web_search.py
  services/
    ingestion.py     ingest_file: parse → chunk → embed → upsert (no DB writes)
    retrieval.py     hybrid search → rerank → top N + confidence signal
    document_pipeline.py  upload validation, dedupe, ingestion jobs, deletion of vectors/files/rows
    chat_turns.py         ChatTurnService: one turn (text or voice) as typed events: save → (router slot) →
                          retrieve → gate → sources → answer; answer length is a parameter; stop = cancel
    sources.py prompts.py language.py   numbered sources + citation checks, answer prompt, answer language
tests/
  integration/       opt-in, real models + Qdrant (RUN_INTEGRATION=1)
```

Each capability has a base class (for example `LLMClient`, `VectorStore`); its methods are added in the phase that builds it. A capability module becomes a package once it grows.
