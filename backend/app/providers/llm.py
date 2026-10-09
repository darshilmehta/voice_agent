"""LLM providers (interface ``LLMClient``, docs/DESIGN.md §6): streamed chat and JSON-schema output.

``OllamaLLM`` talks to Ollama's ``/api/chat`` on the shared HTTP client with ``think`` from the config (off: the
voice loop can't wait for hidden reasoning, §9.1) and per-purpose temperatures (``temperature.answer`` for answers,
``temperature.router`` for JSON decisions).

**One model load for every call (§9.5).** Ollama reloads a model whenever a request asks for another context size,
which costs seconds on this machine. Every request this provider makes (router, answers, titles, summaries, memory,
warm-ups, and any later caller such as the canvas planner) takes ``num_ctx`` and ``keep_alive`` from one place,
``OllamaLLM.runtime_options``; callers can't pass their own. ``warm_up`` reads prompts once (one token each), so the
prompt prefixes the first turns share are already in Ollama's cache.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ValidationError

from .base import HealthStatus, PlaceholderProvider, Provider, ProviderHealth
from .registry import register

LLMRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class LLMMessage:
    role: LLMRole
    content: str


class LLMError(RuntimeError):
    """The model server answered with an error, or its output was unusable (e.g. JSON not matching the schema)."""


class LLMUnavailableError(LLMError):
    """The model server can't be reached (not running, timed out)."""


class LLMClient(Provider):
    capability = "llm"

    def stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """The answer's text as it is generated (chat model and answer temperature unless given)."""
        raise NotImplementedError(f"{type(self).__name__}.stream")

    async def generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """The whole answer at once."""
        parts = [p async for p in self.stream(messages, model=model, temperature=temperature, max_tokens=max_tokens)]
        return "".join(parts)

    async def generate_json[M: BaseModel](
        self,
        messages: Sequence[LLMMessage],
        schema: type[M],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> M:
        """Output constrained to ``schema``'s JSON schema and validated with it (router model and router temperature
        unless given; ``max_tokens`` caps generation). Raises LLMError when the output doesn't validate (also when
        the cap cut it short)."""
        raise NotImplementedError(f"{type(self).__name__}.generate_json")

    async def warm_up(self, prompts: Sequence[Sequence[LLMMessage]], *, model: str | None = None) -> None:
        """Have the model (chat model unless given) read these prompts now, one token each and nothing kept, so a
        model server with a prompt cache has their prefixes ready (the router's and the answers' system prompts, at
        startup). Default: nothing to warm."""


class _ThinkFilter:
    """Drops a leading ``<think>…</think>`` block from streamed text. With ``think: false`` instruct models emit none,
    but some releases leak reasoning into the answer (§9.1); it must never reach the user or the speaker."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._state: Literal["start", "thinking", "text"] = "start"

    def feed(self, piece: str) -> str:
        if self._state == "text":
            return piece
        self._buffer += piece
        if self._state == "start":
            stripped = self._buffer.lstrip()
            if not stripped:
                return ""
            if stripped.startswith(self.OPEN):
                self._state, self._buffer = "thinking", stripped[len(self.OPEN) :]
            elif self.OPEN.startswith(stripped):
                return ""  # could still become "<think>"
            else:
                self._state = "text"
                out, self._buffer = self._buffer, ""
                return out
        end = self._buffer.find(self.CLOSE)
        if end < 0:
            return ""
        out = self._buffer[end + len(self.CLOSE) :].lstrip()
        self._state, self._buffer = "text", ""
        return out

    def flush(self) -> str:
        out = self._buffer if self._state == "start" else ""
        self._buffer = ""
        return out


@register
class OllamaLLM(LLMClient):
    name = "ollama"

    @property
    def base_url(self) -> str:
        return self.config.base_url.rstrip("/")  # type: ignore[attr-defined]

    async def health(self) -> ProviderHealth:
        base = self.base_url
        r, ms, err = await self._probe(f"{base}/api/tags")
        if r is None:
            return self._health(HealthStatus.DOWN, f"Ollama unreachable at {base} ({err})", ms)
        if r.status_code != 200:
            return self._health(HealthStatus.DOWN, f"Ollama returned HTTP {r.status_code}", ms)
        pulled = {m.get("name") for m in r.json().get("models", [])}
        wanted = dict.fromkeys([self.config.chat_model, self.config.router_model])  # type: ignore[attr-defined]
        missing = [m for m in wanted if m not in pulled and f"{m}:latest" not in pulled]
        if missing:
            return self._health(HealthStatus.DEGRADED, f"not pulled: {', '.join(missing)} (ollama pull <model>)", ms)
        return self._health(HealthStatus.OK, f"models ready: {', '.join(wanted)}", ms)

    def runtime_options(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """(options, top-level fields) every request carries: the context size and how long the model stays loaded.
        The only place they are set (module docstring): a request with another ``num_ctx`` would reload the model."""
        cfg = self.config
        options: dict[str, Any] = {"num_ctx": cfg.num_ctx}  # type: ignore[attr-defined]
        fields: dict[str, Any] = {}
        if cfg.keep_alive is not None:  # type: ignore[attr-defined]
            fields["keep_alive"] = cfg.keep_alive  # type: ignore[attr-defined]
        return options, fields

    def request_body(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        temperature: float,
        stream: bool,
        max_tokens: int | None = None,
        format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = self.config
        runtime, fields = self.runtime_options()
        options: dict[str, Any] = {**runtime, "temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "think": cfg.think,  # type: ignore[attr-defined]
            "options": options,
            **fields,
        }
        if format is not None:
            body["format"] = format
        return body

    async def preload(self) -> None:
        """Have Ollama load the chat model now (a request without a prompt only loads it), kept for ``keep_alive``.
        It is loaded with the answers' ``num_ctx``: loaded with Ollama's default context instead, the first answer
        would reload it (measured: first token after 952 ms instead of 136 ms)."""
        cfg = self.config
        runtime, fields = self.runtime_options()
        body: dict[str, Any] = {"model": cfg.chat_model, "options": runtime, **fields}  # type: ignore[attr-defined]
        try:
            response = await self.ctx.http.post(f"{self.base_url}/api/generate", json=body, timeout=self.timeout)
        except httpx.HTTPError as e:
            raise LLMUnavailableError(f"Ollama unreachable at {self.base_url} ({type(e).__name__}: {e})") from e
        if response.status_code != 200:
            raise LLMError(_error_text(response))

    async def warm_up(self, prompts: Sequence[Sequence[LLMMessage]], *, model: str | None = None) -> None:
        """Each prompt read once (one token, the router's temperature), one after another."""
        cfg = self.config
        for messages in prompts:
            await self.generate(
                messages,
                model=model or cfg.chat_model,  # type: ignore[attr-defined]
                temperature=cfg.temperature.router,  # type: ignore[attr-defined]
                max_tokens=1,
            )

    @property
    def timeout(self) -> httpx.Timeout:
        # The shared client's default timeout is the short health-check one; generation needs the configured one.
        return httpx.Timeout(float(self.config.timeout_s), connect=5.0)  # type: ignore[attr-defined]

    async def stream(  # type: ignore[override]
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        cfg = self.config
        body = self.request_body(
            messages,
            model=model or cfg.chat_model,  # type: ignore[attr-defined]
            temperature=cfg.temperature.answer if temperature is None else temperature,  # type: ignore[attr-defined]
            stream=True,
            max_tokens=max_tokens,
        )
        think = _ThinkFilter()
        try:
            async with self.ctx.http.stream(
                "POST", f"{self.base_url}/api/chat", json=body, timeout=self.timeout
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise LLMError(_error_text(response))
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    event = _parse_line(line)
                    if event.get("error"):
                        raise LLMError(f"Ollama: {event['error']}")
                    text = think.feed((event.get("message") or {}).get("content") or "")
                    if text:
                        yield text
                    if event.get("done"):
                        break
        except httpx.TimeoutException as e:
            raise LLMUnavailableError(f"Ollama at {self.base_url} timed out ({type(e).__name__})") from e
        except httpx.HTTPError as e:
            raise LLMUnavailableError(f"Ollama unreachable at {self.base_url} ({type(e).__name__}: {e})") from e
        if rest := think.flush():
            yield rest

    async def generate_json[M: BaseModel](
        self,
        messages: Sequence[LLMMessage],
        schema: type[M],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> M:
        cfg = self.config
        body = self.request_body(
            messages,
            model=model or cfg.router_model,  # type: ignore[attr-defined]
            temperature=cfg.temperature.router if temperature is None else temperature,  # type: ignore[attr-defined]
            stream=False,
            max_tokens=max_tokens,
            format=schema.model_json_schema(),
        )
        try:
            response = await self.ctx.http.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout)
        except httpx.TimeoutException as e:
            raise LLMUnavailableError(f"Ollama at {self.base_url} timed out ({type(e).__name__})") from e
        except httpx.HTTPError as e:
            raise LLMUnavailableError(f"Ollama unreachable at {self.base_url} ({type(e).__name__}: {e})") from e
        if response.status_code != 200:
            raise LLMError(_error_text(response))
        content = ((_parse_line(response.text).get("message") or {}).get("content") or "").strip()
        try:
            return schema.model_validate_json(content)
        except ValidationError as e:
            raise LLMError(
                f"model output doesn't match {schema.__name__}: {e.errors()[:3]}; got {content[:200]!r}"
            ) from e


def _parse_line(line: str) -> dict[str, Any]:
    try:
        event = json.loads(line)
    except json.JSONDecodeError as e:
        raise LLMError(f"Ollama sent invalid JSON: {line[:200]!r}") from e
    if not isinstance(event, dict):
        raise LLMError(f"Ollama sent unexpected data: {line[:200]!r}")
    return event


def _error_text(response: httpx.Response) -> str:
    try:
        detail = response.json().get("error") or response.text
    except (json.JSONDecodeError, AttributeError):
        detail = response.text
    return f"Ollama returned HTTP {response.status_code}: {str(detail)[:300]}"


@register
class OpenAICompatibleLLM(LLMClient, PlaceholderProvider):
    name = "openai_compatible"
