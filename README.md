# voice_agent (poc_gibberlink)

A fully local, ChatGPT voice-mode–style agent for talking with your documents. Upload PDFs, Word or PowerPoint files and ask about them out loud, in English or Hindi. Interrupt the agent, correct yourself or change topic, and it adapts. Answers cite the page they came from.

Everything runs on this machine: local LLM (Ollama), local vector search (Qdrant), local speech recognition and synthesis. The code is structured so each piece can later be swapped for a hosted service through configuration.

> **Status:** Phase −1 complete. Every local component is downloaded and smoke-tested, including a network-off run. Application code starts at phase 0.

## Docs

- [`docs/DESIGN.md`](docs/DESIGN.md) — architecture, decisions, measured results. Source of truth.
- [`docs/blueprint.md`](docs/blueprint.md) — the original idea document the design came from.
- [`config/README.md`](config/README.md) — local vs cloud configuration.

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

## Smoke tests

Each test is a self-contained script with inline dependencies:

```bash
uv run scripts/smoke/02_qdrant.py
```

Run everything with the network off (turn Wi-Fi off first):

```bash
scripts/smoke/run_offline.sh
```

## Layout

```text
config/         local.config.json (default), cloud.config.json (template)
docs/           design + original blueprint
infra/          docker-compose (Qdrant)
scripts/setup/  model downloads
scripts/smoke/  Phase −1 smoke tests (+ web/ browser VAD checks)
data/           models, uploads, databases (git-ignored)
```
