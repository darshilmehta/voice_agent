# Backend image. Build from the repo root:
#   docker build -f infra/docker/backend.Dockerfile -t voice-agent-backend .
# Includes the `ml` dependency group (Docling, BGE-M3, reranker, faster-whisper, Kokoro, Silero VAD) with CPU-only
# PyTorch on Linux (tool.uv.sources in backend/pyproject.toml). Model weights are not baked in: mount data/ (with
# data/models) at /app/data. The ml group needs a lot of memory, see infra/docker-compose.yml.

# ---- build: install the locked dependencies into a virtualenv
FROM python:3.12-slim AS build

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN pip install --no-cache-dir uv==0.12.23

WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock backend/README.md ./
# The cache mount keeps the downloaded wheels (about 1.5 GB with the ml group) out of the image and across rebuilds.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group ml --no-install-project

# ---- runtime: the virtualenv and the app on python:3.12-slim, no build tools
FROM python:3.12-slim

# No system packages: every library the ml group loads ships inside its wheel (libgomp in torch and ctranslate2,
# libsndfile in soundfile, FFmpeg in av, espeak-ng and its data in espeakng-loader, OpenCV headless needs no libGL or
# GLib), and Docling's OCR models come from data/models. image_check.py below fails the build if that stops being true.
RUN useradd --create-home --uid 10001 app

ENV PYTHONUNBUFFERED=1 \
    APP_ROOT_DIR=/app \
    APP_CONFIG_FILE=config/docker.config.json \
    PATH="/app/backend/.venv/bin:$PATH"

WORKDIR /app/backend
COPY --from=build /app/backend/.venv ./.venv
COPY infra/docker/image_check.py /app/image_check.py
RUN python /app/image_check.py

COPY backend/app ./app
COPY config /app/config
RUN mkdir -p /app/data && chown app /app/data

USER app
EXPOSE 8000
# Models load in the background after startup, so /health answers at once; the long start period is for slow disks.
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["python", "-m", "app"]
