"""Application configuration.

One JSON file configures the backend (docs/DESIGN.md §6):

    APP_CONFIG_FILE (default config/local.config.json, relative to the project root)
      → parse JSON
      → resolve "${VAR}" strings from the environment (any missing variable is an error)
      → apply env overrides, nested with "__"   (LLM__CHAT_MODEL=qwen3:8b)
      → validate (unknown keys are rejected, so the file and this schema can't drift)

Provider names are validated later by the registry, which knows what is implemented.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILE = "config/local.config.json"

Language = Literal["en", "hi"]
TorchDevice = Literal["cpu", "mps", "cuda"]


class ConfigError(RuntimeError):
    """Configuration is missing, malformed or inconsistent. Raised before the app starts."""


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AppSection(Section):
    name: str
    environment: Literal["development", "test", "production"]
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    log_format: Literal["console", "json"]
    data_dir: str


class ServerSection(Section):
    host: str
    port: int = Field(ge=1, le=65535)
    public_base_url: str
    cors_allowed_origins: list[str]
    trusted_proxy_ips: list[str]
    max_upload_mb: int = Field(gt=0)
    request_timeout_s: int = Field(gt=0)


class AuthSection(Section):
    provider: str
    issuer_url: str | None
    audience: str | None
    client_id: str | None
    jwks_url: str | None


class ClientSection(Section):
    app_title: str
    languages: list[Language] = Field(min_length=1)
    default_language: Language
    voice_enabled: bool
    show_debug_panel: bool

    @model_validator(mode="after")
    def _default_language_is_offered(self) -> ClientSection:
        if self.default_language not in self.languages:
            raise ValueError(f"default_language {self.default_language!r} is not in languages {self.languages}")
        return self


class Temperatures(Section):
    router: float = Field(ge=0, le=2)
    answer: float = Field(ge=0, le=2)


class LLMSection(Section):
    provider: str
    base_url: str
    api_key: str | None
    chat_model: str
    router_model: str
    think: bool
    num_ctx: int = Field(gt=0)
    keep_alive: str | None
    timeout_s: int = Field(gt=0)
    temperature: Temperatures

    @model_validator(mode="after")
    def _no_hosted_ollama_models(self) -> LLMSection:
        # Ollama's "-cloud" tags run on Ollama's servers, which would silently break the local-only promise.
        if self.provider == "ollama":
            hosted = [m for m in (self.chat_model, self.router_model) if m.endswith("-cloud") or ":cloud" in m]
            if hosted:
                raise ValueError(f"Ollama cloud-hosted models are not allowed: {hosted}")
        return self


class EmbeddingsSection(Section):
    provider: str
    model: str
    device: TorchDevice
    fp16: bool
    batch_size: int = Field(gt=0)
    base_url: str | None
    api_key: str | None


class RerankerSection(Section):
    provider: str
    model: str
    device: TorchDevice
    fp16: bool
    base_url: str | None
    api_key: str | None


class VectorStoreSection(Section):
    provider: str
    url: str
    api_key: str | None
    collection: str
    prefer_grpc: bool
    timeout_s: int = Field(gt=0)


class MetadataDBSection(Section):
    provider: str
    url: str
    pool_size: int | None


class ObjectStoreSection(Section):
    provider: str
    root: str | None
    bucket: str | None
    region: str | None
    endpoint_url: str | None
    access_key_id: str | None
    secret_access_key: str | None


class ChunkingSection(Section):
    version: str
    target_tokens: int = Field(gt=0)
    overlap_tokens: int = Field(ge=0)


class IngestionSection(Section):
    provider: str
    artifacts_path: str
    ocr: bool
    # "auto" is deliberately not allowed: docling's auto mode silently skips OCR when it finds no engine (§9.4).
    ocr_engine: Literal["rapidocr", "easyocr", "tesseract", "ocrmac"]
    max_pages: int = Field(gt=0)
    allowed_extensions: list[str]
    chunking: ChunkingSection


class RetrievalSection(Section):
    prefetch_k: int = Field(gt=0)
    fusion: Literal["rrf", "dbsf"]
    rerank_top_n: int = Field(gt=0)
    min_rerank_score: float = Field(ge=0, le=1)
    context_token_budget: int = Field(gt=0)


class STTSection(Section):
    provider: str
    model: str
    device: Literal["cpu", "gpu", "cuda"]
    compute_type: str | None
    languages: list[Language]
    base_url: str | None
    api_key: str | None


class VADSection(Section):
    provider: str
    threshold: float = Field(gt=0, lt=1)
    min_speech_ms: int = Field(ge=0)
    end_of_turn_ms: int = Field(gt=0)


class TTSSection(Section):
    provider: str
    model: str
    device: TorchDevice
    voices: dict[Language, str]
    sample_rate: int = Field(gt=0)
    base_url: str | None
    api_key: str | None


class BargeInSection(Section):
    duck_volume: float = Field(ge=0, le=1)
    backchannel_max_words: int = Field(ge=0)
    decision_timeout_ms: int = Field(gt=0)


class VoiceSection(Section):
    max_spoken_sentences: int = Field(gt=0)
    barge_in: BargeInSection


class IceServer(Section):
    urls: list[str]
    username: str | None = None
    credential: str | None = None


class AudioTransportSection(Section):
    provider: str
    input_sample_rate: int = Field(gt=0)
    ice_servers: list[IceServer]


class WebSearchSection(Section):
    enabled: bool
    provider: str
    url: str | None
    api_key: str | None
    max_results: int = Field(gt=0)
    timeout_s: int = Field(gt=0)
    stream_partial_results: bool


class ToolsSection(Section):
    web_search: WebSearchSection


class JobQueueSection(Section):
    provider: str
    url: str | None
    concurrency: int = Field(gt=0)


class SessionStoreSection(Section):
    provider: str
    url: str | None
    ttl_s: int = Field(gt=0)


class EventBusSection(Section):
    provider: str
    url: str | None


class ModelCacheSection(Section):
    hf_home: str


class ObservabilitySection(Section):
    metrics: Literal["none", "prometheus"]
    tracing: Literal["none", "otlp"]
    otlp_endpoint: str | None


class Settings(Section):
    profile: Literal["local", "cloud"]
    strict_offline: bool
    strict_offline_exceptions: list[Literal["web_search"]]
    # Hostnames that resolve to this machine without being loopback, e.g. Docker Compose service names and
    # host.docker.internal. Top-level, so (like strict_offline itself) only the config file can set it.
    strict_offline_local_hosts: list[str]
    app: AppSection
    server: ServerSection
    auth: AuthSection
    client: ClientSection
    llm: LLMSection
    embeddings: EmbeddingsSection
    reranker: RerankerSection
    vector_store: VectorStoreSection
    metadata_db: MetadataDBSection
    object_store: ObjectStoreSection
    ingestion: IngestionSection
    retrieval: RetrievalSection
    stt: STTSection
    vad: VADSection
    tts: TTSSection
    voice: VoiceSection
    audio_transport: AudioTransportSection
    tools: ToolsSection
    job_queue: JobQueueSection
    session_store: SessionStoreSection
    event_bus: EventBusSection
    model_cache: ModelCacheSection
    observability: ObservabilitySection

    @model_validator(mode="after")
    def _local_hosts_are_bare_hostnames(self) -> Settings:
        bad = [h for h in self.strict_offline_local_hosts if not re.fullmatch(r"[A-Za-z0-9.-]+", h)]
        if bad:
            raise ValueError(f"strict_offline_local_hosts must be bare hostnames (no scheme, port or path): {bad}")
        return self

    # Set by the loader, not read from the file.
    root_dir: Path = Field(default=PROJECT_ROOT, exclude=True)
    source: Path | None = Field(default=None, exclude=True)

    def path(self, value: str | Path) -> Path:
        """Resolve a configured path: absolute paths as-is, relative ones against the project root."""
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self.root_dir / p).resolve()


# ------------------------------------------------------------------ loading

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _interpolate(node: Any, env: Mapping[str, str], missing: set[str]) -> Any:
    if isinstance(node, str):

        def sub(m: re.Match[str]) -> str:
            name = m.group(1)
            if name not in env:
                missing.add(name)
                return m.group(0)
            return env[name]

        return _VAR.sub(sub, node)
    if isinstance(node, list):
        return [_interpolate(v, env, missing) for v in node]
    if isinstance(node, dict):
        return {k: _interpolate(v, env, missing) for k, v in node.items()}
    return node


def _parse_override(value: str) -> Any:
    """Env values are JSON when they parse as JSON (8192, true, null, ["en"]), plain strings otherwise."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _apply_overrides(raw: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    """Apply SECTION__KEY[__SUBKEY]=value variables whose first segment is a top-level config key."""
    for name, value in sorted(env.items()):
        if "__" not in name:
            continue
        parts = [p.lower() for p in name.split("__")]
        if parts[0] not in raw or not all(parts):
            continue
        node = raw
        for i, key in enumerate(parts[:-1]):
            if not isinstance(node.get(key), dict):
                raise ConfigError(f"env override {name}: {'.'.join(parts[: i + 1])} is not a section")
            node = node[key]
        if parts[-1] not in node:
            raise ConfigError(f"env override {name}: unknown key {'.'.join(parts)}")
        node[parts[-1]] = _parse_override(value)
    return raw


def _format_validation_error(err: ValidationError, source: Path) -> str:
    lines = [f"invalid configuration in {source}:"]
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"])
        lines.append(f"  - {loc}: {e['msg']}")
    return "\n".join(lines)


def load_settings(path: str | Path | None = None, env: Mapping[str, str] | None = None) -> Settings:
    """Load, interpolate, override and validate the configuration. Raises ConfigError with every problem found."""
    env = dict(os.environ if env is None else env)
    root = Path(env.get("APP_ROOT_DIR") or PROJECT_ROOT).resolve()
    source = Path(path or env.get("APP_CONFIG_FILE") or DEFAULT_CONFIG_FILE).expanduser()
    if not source.is_absolute():
        source = root / source

    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {source}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"config file is not valid JSON: {source}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"config file must contain a JSON object: {source}")

    missing: set[str] = set()
    raw = _interpolate(raw, env, missing)
    if missing:
        raise ConfigError(f"{source} references unset environment variables: {', '.join(sorted(missing))}")
    raw = _apply_overrides(raw, env)

    try:
        settings = Settings.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(_format_validation_error(e, source)) from e
    return settings.model_copy(update={"root_dir": root, "source": source})
