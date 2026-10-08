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

## Run (natively, recommended on a Mac)

Backend on `127.0.0.1:8000` ([`backend/README.md`](backend/README.md)):

```bash
cd backend && uv sync && uv run python -m app
```

Frontend on `127.0.0.1:3000` ([`frontend/README.md`](frontend/README.md)):

```bash
cd frontend && npm install && npm run dev
```

Open http://localhost:3000. The status panel shows every component; the same check from the terminal:

```bash
curl -s localhost:8000/health
```

## Run everything in Docker

Qdrant, backend and frontend in containers (Ollama stays on the host); see [`config/README.md`](config/README.md):

```bash
docker compose -f infra/docker-compose.yml --profile full up -d --build
```

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
