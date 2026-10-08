# Backend image. Build from the repo root:
#   docker build -f infra/docker/backend.Dockerfile -t voice-agent-backend .
# Models are not baked in: mount data/ (with data/models) at /app/data.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    APP_ROOT_DIR=/app \
    APP_CONFIG_FILE=config/docker.config.json

RUN pip install --no-cache-dir uv==0.12.23 \
 && useradd --create-home --uid 10001 app

WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock backend/README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY backend/app ./app
COPY config /app/config
RUN mkdir -p /app/data && chown app /app/data

ENV PATH="/app/backend/.venv/bin:$PATH"
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["python", "-m", "app"]
