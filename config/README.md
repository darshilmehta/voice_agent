# Configuration

One JSON file configures the whole backend. The frontend gets its settings from the backend at runtime.

| File | Used by | Purpose |
|---|---|---|
| `local.config.json` | default | Everything on this machine. Fields only clouds need are `null`. |
| `cloud.config.json` | nothing (template) | Example server deployment with placeholder hosts (`*.example.com`) and `${VAR}` secrets. Copy it, edit it, point the app at it. |
| `docker.config.json` | the backend container (Compose profile `full`) | The local setup inside Docker: Qdrant at `qdrant:6333`, Ollama on the host at `host.docker.internal:11434`, CPU devices, faster-whisper (mlx-whisper is macOS-only). |

Both files share exactly the same keys and are validated by the same schema, so the template can't silently drift from the code (a test loads both).

## How the backend loads config

```text
APP_CONFIG_FILE              (default: config/local.config.json)
  → parse JSON
  → resolve "${VAR}" strings from environment variables   (missing var = startup error)
  → apply env overrides, nested with "__"                 (e.g. LLM__CHAT_MODEL=qwen3:4b-instruct)
  → validate (pydantic)
  → strict_offline check
  → build providers
```

- **Secrets never go in the file.** Use `"${VAR}"` placeholders and set the variables in the deployment environment (or a `.env` that is never committed).
- **Top-level switches are file-only.** `profile`, `strict_offline`, `strict_offline_exceptions` and `strict_offline_local_hosts` can't be changed by environment variables, so the offline guard can only be relaxed by editing the file.
- **`strict_offline_local_hosts`** lists hostnames that are on this machine without being loopback: Docker Compose service names and `host.docker.internal` (exact names only; `docker.config.json` uses `["qdrant", "host.docker.internal"]`).
- **`ingestion.ocr_engine` must name an engine** (`rapidocr`, …). Docling's `auto` mode silently skips OCR when it finds none, so it is rejected.
- **`strict_offline: true`** (local) makes startup fail if any provider is remote or any URL is not loopback, and forces Hugging Face offline mode. The only escape hatch is `strict_offline_exceptions` (e.g. `["web_search"]`), which allows exactly that capability to reach the network.

## Running everything in Docker

```bash
docker compose -f infra/docker-compose.yml --profile full up -d --build
```

Qdrant, backend and frontend run in containers (ports published on `127.0.0.1` only); Ollama stays on the host. Models are mounted from `data/`, never baked into images. On macOS Docker has no Apple GPU, so models run on CPU: once the backend loads models (phase 1 onwards) raise Docker Desktop's memory limit accordingly (BGE-M3 alone needs about 2.3 GB on CPU) — or keep running natively, which is the recommended way on a Mac. On Linux, start Ollama with `OLLAMA_HOST=0.0.0.0` so the container can reach it.

## Deploying to a server

```bash
cp config/cloud.config.json config/prod.config.json   # edit hosts, buckets, models
export APP_CONFIG_FILE=config/prod.config.json
export LLM_API_KEY=… QDRANT_API_KEY=… DB_PASSWORD=… S3_ACCESS_KEY_ID=… S3_SECRET_ACCESS_KEY=… REDIS_PASSWORD=… TURN_PASSWORD=…
docker compose -f infra/docker-compose.yml --profile full up -d
```

Frontend: set one variable, `BACKEND_URL` (e.g. `https://api.gibberlink.example.com`). It is read at runtime, so the same frontend image works for any environment. Everything else the UI needs (languages, voice on/off, auth client id) comes from the backend's `GET /api/config/public`, built from the `client` and `auth` sections.

## Provider status

What the POC builds vs. what stays a placeholder. (No application code exists yet — see `docs/DESIGN.md` §10 for build phases.)

| Capability | Built in this POC | Placeholder (raises NotImplementedError) |
|---|---|---|
| llm | `ollama` | `openai_compatible` |
| embeddings | `bge_m3` (mps / cuda / cpu) | — |
| reranker | `bge_reranker` (mps / cuda / cpu) | — |
| vector_store | `qdrant` (local or remote URL + api_key) | — |
| metadata_db | `sqlite` | `postgres` |
| object_store | `filesystem` | `s3` |
| ingestion | `docling` | — |
| stt | `mlx_whisper` (macOS), `faster_whisper` (cpu / cuda) | — |
| vad | `silero` | — |
| tts | `kokoro` | — |
| audio_transport | `websocket` | `webrtc` |
| auth | `none` | `oidc` |
| job_queue / session_store / event_bus | `in_process` / `in_memory` | `redis` |
| observability | `none` | `prometheus`, `otlp` |
| tools.web_search | `searxng` (local; off until phase 8) | `search_api` (production) |

So `cloud.config.json` as written would start only once the placeholder providers it names are implemented. The self-hosted pieces (Qdrant, BGE-M3, reranker, faster-whisper, Kokoro, Docling on a CUDA server) already work with config changes alone.
