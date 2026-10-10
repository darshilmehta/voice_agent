# Configuration

One JSON file configures the whole backend of Docent ("Talk to your documents"). `client.app_title` is the name the UI and the OpenAPI docs show ("Docent"); the config keys, `app.name` (`poc_gibberlink`) and paths are internal identifiers. The frontend gets its settings from the backend at runtime.

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

Qdrant, backend and frontend run in containers (ports published on `127.0.0.1` only); Ollama stays on the host. Models are mounted from `data/` (`/app/data/models` in the container, which is where `docker.config.json`'s relative `data/...` paths resolve with `APP_ROOT_DIR=/app`), never baked into images: run `scripts/setup/download_models.sh all` first. The backend image installs the `ml` group (Docling, BGE-M3, reranker, faster-whisper, Kokoro, Silero) with CPU-only PyTorch: on Linux `backend/pyproject.toml` takes `torch` and `torchvision` from PyTorch's CPU index, which leaves out several GB of CUDA libraries. On a CUDA server remove those two `tool.uv.sources` lines and run `uv lock` (and set the devices in your config to `cuda`). On Linux, start Ollama with `OLLAMA_HOST=0.0.0.0` so the container can reach it.

**Memory.** On macOS Docker has no Apple GPU, so every model runs on the CPU, and the backend loads all of them at startup. Estimates: the process baseline and faster-whisper were measured in the container, the other models come from `docs/DESIGN.md` §9 (BGE-M3 and the reranker are 568M-parameter models in fp32 on the CPU).

| Process | Estimated memory |
|---|---|
| BGE-M3 embedder | ~2.3 GB |
| bge-reranker-v2-m3 | ~2.3 GB |
| faster-whisper small (int8) | ~0.4 GB (0.65 GB while loading) |
| Kokoro-82M | ~0.55 GB |
| Silero VAD | < 0.1 GB |
| Docling layout, table and OCR models | ~1.7 GB peak, only while a document is being ingested |
| Python, torch, transformers, FastAPI | ~0.7 GB |
| **Backend total** | **~6.5 GB steady, ~8 GB while ingesting** |
| Qdrant / frontend | ~0.3 GB / ~0.15 GB |

Give Docker at least 10 GB (Ollama's model, about 3 GB for `qwen3:4b-instruct`, runs on the host in addition). The Compose file caps the backend container at `BACKEND_MEM_LIMIT` (default `10g`, no swap) and the frontend at 512 MB. A Docker Desktop VM capped at 1.5 GB, as on the development Mac, can't hold this: models fail to load, and the VM's out-of-memory killer may end any container, Qdrant included. **On a Mac run the backend and frontend natively, which is the recommended way** (`README.md`, "Run"): it needs less memory and uses the Apple GPU (MPS, mlx-whisper). The `full` profile is for Linux servers and larger machines. Docker Desktop's memory setting is yours to change; nothing in this repo does.

## Deploying to a server

```bash
cp config/cloud.config.json config/prod.config.json   # edit hosts, buckets, models
export APP_CONFIG_FILE=config/prod.config.json
export LLM_API_KEY=… QDRANT_API_KEY=… DB_PASSWORD=… S3_ACCESS_KEY_ID=… S3_SECRET_ACCESS_KEY=… REDIS_PASSWORD=… TURN_PASSWORD=…
docker compose -f infra/docker-compose.yml --profile full up -d
```

Frontend: set one variable, `BACKEND_URL` (e.g. `https://api.gibberlink.example.com`). It is read at runtime, so the same frontend image works for any environment. Everything else the UI needs (languages, voice on/off, auth client id) comes from the backend's `GET /api/config/public`, built from the `client` and `auth` sections.

## Live web search (optional)

Questions that need current data ("how is the stock doing today?") can be answered from the documents plus a web search (docs/DESIGN.md §3.7). It is **on** in `local.config.json` (`python3 start.py` starts SearXNG when it is) and off in the Docker config: it is the one feature that sends something off the machine, namely the short English search query (never document text or the transcript; see `backend/README.md`, "Live data"). Locally it goes through a self-hosted SearXNG, which forwards the query to public search engines (they see the query and this machine's IP, no cookies or account).

To turn it on by hand (what `start.py` does), or in another config:

1. Start SearXNG (Compose profile `websearch`; pinned image, `127.0.0.1:8888` only, 256 MB):

   ```bash
   export SEARXNG_SECRET=$(openssl rand -hex 24)   # optional: unset, a random secret is generated at each start
   docker compose -f infra/docker-compose.yml --profile websearch up -d searxng
   curl -s http://127.0.0.1:8888/healthz          # OK
   ```

   Its settings are `infra/searxng/settings.yml` (JSON output on, limiter off for this single local client, engines: DuckDuckGo, Bing, Brave, Yahoo, Wikipedia and the news engines). Stop it with `docker compose -f infra/docker-compose.yml --profile websearch rm -sf searxng`. Don't use `--profile websearch down`: `down` acts on the whole project and would also remove the Qdrant container.

2. In the config file (top-level keys are file-only), allow the capability through the offline guard and switch the tool on:

   ```json
   "strict_offline_exceptions": ["web_search"],
   "tools": { "web_search": { "enabled": true, … } }
   ```

   (`TOOLS__WEB_SEARCH__ENABLED=true` works too, but the exception must be in the file: with `strict_offline: true` and no exception, startup fails.) Restart the backend; `/health` reports `web_search` as `ok` (or `degraded` naming configured engines SearXNG doesn't have) and `GET /api/config/public` has `features.web_search: true`. The voice filler ("Let me look that up.") is synthesized at startup.

| `tools.web_search` key | Local value | Meaning |
|---|---|---|
| `enabled` | `false` | The tool runs only when true (and allowed by the offline guard). |
| `provider` | `searxng` | `search_api` is the production placeholder (Brave / Tavily / Exa …), not implemented. |
| `url` / `api_key` | `http://127.0.0.1:8888` / `null` | SearXNG's address (Docker: `http://searxng:8080`); the key is for a search API. |
| `max_results` | `5` | Results per search (1–10), across all engines. |
| `timeout_s` | `5` | The whole search. No results by then: the answer says live data couldn't be fetched and answers from the documents. |
| `stream_partial_results` | `true` | Start the answer on the first results and continue with later ones; `false` waits for the whole search. |
| `engines` | `["duckduckgo", "duckduckgo news", "bing", "bing news", "brave", "yahoo", "wikipedia"]` | One SearXNG request per engine, so the fastest engine answers first. Engines SearXNG doesn't have are skipped. `[]`: one request with SearXNG's defaults. |
| `request_timeout_s` | `3.5` | One engine request. |
| `partial_wait_ms` | `150` | After the first result, how long to wait for other engines before the answer starts. |
| `max_continuations` | `1` | Short continuations (one more model call each) for results that arrive after the answer started; `0`: none. |
| `fetch_pages` | `1` | Top result pages fetched (public addresses only, ≤ 400 KB, text only) for the continuation; `0`: snippets only, nothing fetched besides the search. |

Public engines rate-limit and block automated traffic (CAPTCHAs, "too many requests"), so some engines often return nothing; that is why several are asked in parallel. Fine for a demo, not for production (DESIGN §3.7).

## Live visual canvas

Tables are typed into datasets after ingestion, and each project gets an overview (DESIGN §12.1). The visual planner is one JSON call to `llm.router_model`.

| `canvas` key | Local value | Meaning |
|---|---|---|
| `planner_timeout_ms` | `8000` | The planner's model call. Past it, a visual the user asked for ("show me …") gets the best table's default chart; otherwise none. Ollama answers one request at a time, so the planner may wait behind the spoken answer. |
| `planner_max_tokens` | `200` | Output cap of that call (a plan is ~60 tokens of compact JSON). |
| `planner_candidates` | `4` | Datasets offered to the planner per question (best matches of the chat's documents). |
| `max_panels` | `12` | Panels per chat canvas; adding one more removes the oldest unpinned panel. |
| `overview_panels` | `3` | Panels on a project's overview (KPI tiles, a trend, a composition); `0`: no overview. |

## Noisy rooms

`voice.noise` (DESIGN §3.10 "Noisy rooms", measured in §9.7). Levels are dBFS (10·log10 of the mean square of 32 ms frames); the browser gets its share as `voice_input` in `/api/config/public`.

| `voice.noise` key | Local value | Meaning |
|---|---|---|
| `denoise` | `rnnoise` | The browser's denoiser on the microphone (`off`: none). |
| `adaptive_gating` | `true` | The noise-floor gate on both VADs, and turns announced only near the user's level. |
| `floor_window_ms`, `floor_percentile` | `8000`, `20` | The noise floor: this percentile of the last window of frame levels. |
| `start_snr_db`, `end_snr_db` | `9`, `4` | A turn opens only this far above the floor; within `end_snr_db` of it is silence. |
| `quiet_floor_dbfs`, `loud_floor_dbfs` | `-60`, `-35` | Between these floors the VAD threshold rises from `vad.threshold` to `noisy_threshold` (`0.8`) and the minimum speech from `vad.min_speech_ms` to `noisy_min_speech_ms` (`400`). |
| `drop_background_speech` | `true` | Drop speech that wasn't said to the agent, silently (`false`: every transcript is answered or asked again). |
| `assumed_user_dbfs`, `assumed_margin_db` | `-26`, `2` | The user's level before their first answered turn, and the extra room it gets. |
| `far_field_db`, `far_field_hard_db` | `6`, `8` | Below the user's level by `far_field_hard_db`: dropped (and never announced); by `far_field_db`: dropped if unsure, a fragment or weak. |
| `close_db` | `6` | Within this of the user's level, weak-against-the-floor speech is still theirs (a loud café). |
| `min_snr_db` | `6` | "Weak against the floor". |
| `unsure_avg_logprob`, `unsure_no_speech_prob` | `-0.65`, `0.5` | "Unsure": Whisper's confidence. |
| `short_fragment_words` | `3` | "A fragment": this many words or fewer with no content word. |

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
| tools.web_search | `searxng` (local; on in local.config.json, see "Live web search") | `search_api` (production) |

So `cloud.config.json` as written would start only once the placeholder providers it names are implemented. The self-hosted pieces (Qdrant, BGE-M3, reranker, faster-whisper, Kokoro, Docling on a CUDA server) already work with config changes alone.
