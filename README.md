# voice_agent (poc_gibberlink)

A fully local, ChatGPT voice-mode–style agent for talking with your documents. Upload PDFs, Word or PowerPoint files and ask about them out loud, in English or Hindi. Interrupt the agent, correct yourself or change topic, and it adapts. Answers cite the page they came from.

Everything runs on this machine: local LLM (Ollama), local vector search (Qdrant), local speech recognition and synthesis. The code is structured so each piece can later be swapped for a hosted service through configuration.

> **Status:** Phase 0 complete: config, provider registry with health checks, backend API, frontend shell, Docker images, CI. Phase 1 (projects, document ingestion, text chat with citations) is in progress.

## Docs

- [`docs/DESIGN.md`](docs/DESIGN.md) — architecture, decisions, measured results. Source of truth.
- [`docs/blueprint.md`](docs/blueprint.md) — the original idea document the design came from.
- [`config/README.md`](config/README.md) — local, Docker and cloud configuration.
- [`backend/README.md`](backend/README.md), [`frontend/README.md`](frontend/README.md) — running and testing each part.
- [`docs/prototypes/voice-presence.html`](docs/prototypes/voice-presence.html) — the voice UI prototype (§3.8).

## Stack

| Role | Local choice |
|---|---|
| LLM (router + answers) | `qwen3:4b-instruct` via Ollama |
| Embeddings / reranker | BAAI/bge-m3 / BAAI/bge-reranker-v2-m3 |
| Vector store | Qdrant (Docker, loopback only) |
| Document parsing | Docling + RapidOCR |
| Speech to text | mlx-whisper small (Apple GPU) |
| Text to speech | Kokoro-82M |
| Voice activity | Silero VAD (browser + server) |

## Prerequisites (macOS, Apple Silicon)

```bash
brew install ollama uv espeak-ng
```

```bash
brew services start ollama && ollama pull qwen3:4b-instruct
```

```bash
docker compose -f infra/docker-compose.yml up -d qdrant
```

```bash
scripts/setup/download_models.sh all
```

Docker Desktop, Python 3.12, Node 24 and ffmpeg are also expected.

## Quick start (one command)

After the prerequisites are installed, from the repo root:

```bash
python3 start.py
```

It starts what is missing, in order, and skips what is already running:
- **Ollama:** started with its prompt cache capped (`LLAMA_ARG_CACHE_RAM`, DESIGN §8). If the cap is unset, it is set and Ollama is restarted. The chat model is pulled if missing.
- **Qdrant:** Docker Desktop and the Qdrant container are started.
- **SearXNG:** started too when the config turns live web search on (it is on in `config/local.config.json`).
- **Model weights:** you are offered the download if any are missing.
- **Backend:** waits until `preload: ready`.
- **Frontend:** started next.
- **Browser:** Chrome opens at http://localhost:3000.

The backend's and frontend's output is shown in that terminal and saved in `data/logs/`. **Ctrl+C**, or closing that terminal, stops both, while Qdrant, SearXNG and Ollama keep running. The other commands:

```bash
python3 start.py status
```

```bash
python3 start.py stop
```

`stop` works from any terminal and stops everything except Ollama: the backend and frontend (also ones left running by a `start.py` that is gone), then SearXNG and Qdrant. When Docker isn't running, it says so. The start options are `--stop-qdrant` (Ctrl+C also stops Qdrant and SearXNG), `--no-browser`, `-y` (accept downloads) and `--no-ollama-cap`. The steps below do the same by hand.

## Run (natively, recommended on a Mac)

Backend on `127.0.0.1:8000` ([`backend/README.md`](backend/README.md)):

```bash
cd backend && uv sync --group ml && uv run python -m app
```

`--group ml` installs the local models' libraries (Docling, BGE-M3, the reranker, Whisper, Kokoro, Silero); without it the app starts but documents, retrieval and voice are unavailable (`/health` marks them `degraded`). The model weights come from `scripts/setup/download_models.sh all`, and Qdrant (`docker compose -f infra/docker-compose.yml up -d qdrant`) and Ollama with `qwen3:4b-instruct` must be running. The models preload for ~20–35 s after startup (`/health` → `preload: ready`).

Frontend on `127.0.0.1:3000` ([`frontend/README.md`](frontend/README.md)):

```bash
cd frontend && npm install && npm run dev
```

Open http://localhost:3000. The status panel shows every component; the same check from the terminal:

```bash
curl -s localhost:8000/health
```

## Run everything in Docker

Qdrant, backend and frontend in containers; Ollama stays on the host. The backend image includes the `ml` dependency group with CPU-only PyTorch (no CUDA libraries) but no model weights: run `scripts/setup/download_models.sh all` first, and `data/models` is mounted into the container. Details in [`config/README.md`](config/README.md):

```bash
docker compose -f infra/docker-compose.yml --profile full up -d --build
```

**This needs far more memory than the 1.5 GB Docker Desktop cap this project is developed under (only Qdrant fits there).** The backend loads every model on the CPU at startup. Estimates (the process baseline and faster-whisper were measured in the container, the rest come from `docs/DESIGN.md` §9):

| Process | Memory |
|---|---|
| BGE-M3 embedder | ~2.3 GB |
| bge-reranker-v2-m3 | ~2.3 GB |
| faster-whisper small (int8) | ~0.4 GB |
| Kokoro-82M | ~0.55 GB |
| Docling models | ~1.7 GB peak, only while a document is ingested |
| Python, torch, transformers, FastAPI | ~0.7 GB |
| **Backend** | **~6.5 GB steady, ~8 GB while ingesting** (capped at 10 GB by `BACKEND_MEM_LIMIT`) |
| Qdrant, frontend | ~0.3 GB, ~0.15 GB |

Give Docker at least 10 GB, and Ollama (about 3 GB for `qwen3:4b-instruct`) is on top. With less, the models fail to load or the out-of-memory killer ends processes in Docker's VM, Qdrant included. **On a Mac, run natively** (above): it needs less and uses the Apple GPU. Don't run the containers and the native backend/frontend together; they use the same ports.

## Voice UI prototype

From the project root:

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

Then open http://localhost:8765/docs/prototypes/voice-presence.html.

## Smoke tests

Each test is a self-contained script with inline dependencies:

```bash
uv run scripts/smoke/02_qdrant.py
```

Run everything with the network off (turn Wi-Fi off first):

```bash
scripts/smoke/run_offline.sh
```

## Development

Every change goes through a pull request into `main`. CI runs on each PR: backend lint + tests, frontend build + typecheck, and Docker image builds.

```bash
cd backend && uv run ruff check . && uv run pytest
```

```bash
cd frontend && npm run build && npm run typecheck
```

## Layout

```text
start.py        one-command start of everything (Quick start)
backend/        FastAPI app: config, offline guard, providers, API, tests
frontend/       Next.js app shell
config/         local.config.json (default), docker.config.json, cloud.config.json (template)
docs/           design, original blueprint, UI prototypes
infra/          docker-compose (Qdrant; profile "full" adds backend + frontend), Dockerfiles
scripts/setup/  model downloads
scripts/smoke/  Phase −1 smoke tests (+ web/ browser VAD checks)
.github/        CI workflow
data/           models, uploads, databases (git-ignored)
```
