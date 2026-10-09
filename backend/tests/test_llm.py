"""OllamaLLM over a mocked HTTP transport: request body, streaming, JSON-schema output, errors."""

from __future__ import annotations

import json
from typing import Literal

import httpx
import pytest
from pydantic import BaseModel

from app.providers.base import ProviderContext
from app.providers.llm import LLMError, LLMMessage, LLMUnavailableError, OllamaLLM, _ThinkFilter

MESSAGES = [LLMMessage("system", "Be brief."), LLMMessage("user", "What was the EBITDA margin?")]


def ndjson(*events: dict) -> bytes:
    return b"".join(json.dumps(e).encode() + b"\n" for e in events)


def chunks(*texts: str) -> bytes:
    events = [{"message": {"role": "assistant", "content": t}, "done": False} for t in texts]
    return ndjson(*events, {"message": {"role": "assistant", "content": ""}, "done": True, "eval_count": 9})


class Recorder:
    def __init__(self, respond) -> None:
        self.respond = respond
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.respond(request)

    def body(self, i: int = -1) -> dict:
        return json.loads(self.requests[i].content)


def make_llm(load_local, respond, **env: str) -> tuple[OllamaLLM, Recorder]:
    settings = load_local(**env)
    recorder = Recorder(respond)
    http = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    return OllamaLLM(settings.llm, ProviderContext(settings=settings, http=http)), recorder


async def collect(llm: OllamaLLM, **kw) -> list[str]:
    return [p async for p in llm.stream(MESSAGES, **kw)]


async def test_stream_sends_config_and_yields_text(load_local):
    llm, rec = make_llm(load_local, lambda r: httpx.Response(200, content=chunks("EBITDA margin ", "was 18.2% [S1].")))
    assert await collect(llm, max_tokens=256) == ["EBITDA margin ", "was 18.2% [S1]."]
    req = rec.requests[0]
    assert req.method == "POST" and req.url.path == "/api/chat"
    body = rec.body()
    assert body["model"] == "qwen3:4b-instruct" and body["stream"] is True and body["think"] is False
    assert body["keep_alive"] == "30m"
    assert body["options"] == {"num_ctx": 8192, "temperature": 0.3, "num_predict": 256}  # answer temperature
    assert body["messages"] == [{"role": m.role, "content": m.content} for m in MESSAGES]
    assert "format" not in body


async def test_overrides_and_null_keep_alive(load_local):
    llm, rec = make_llm(load_local, lambda r: httpx.Response(200, content=chunks("ok")), LLM__KEEP_ALIVE="null")
    assert await llm.generate(MESSAGES, model="qwen3:8b", temperature=0.0) == "ok"
    body = rec.body()
    assert body["model"] == "qwen3:8b" and body["options"]["temperature"] == 0.0 and "keep_alive" not in body
    assert "num_predict" not in body["options"]


async def test_leaked_reasoning_never_reaches_the_answer(load_local):
    leaked = chunks("<thi", "nk>the user wants", " x</think>\n\n", "18.2%")
    llm, _ = make_llm(load_local, lambda r: httpx.Response(200, content=leaked))
    assert "".join(await collect(llm)) == "18.2%"


@pytest.mark.parametrize(
    ("pieces", "expected"),
    [
        (["Hello ", "world"], "Hello world"),
        (["<b>bold</b>"], "<b>bold</b>"),
        (["<think about it"], "<think about it"),
        (["  ", "<think>", "plan</think>", "Answer"], "Answer"),
        (["<thi"], "<thi"),  # flushed at the end
    ],
)
def test_think_filter(pieces, expected):
    f = _ThinkFilter()
    out = "".join(f.feed(p) for p in pieces) + f.flush()
    assert out == expected


async def test_http_error_and_error_events_raise_llm_error(load_local):
    llm, _ = make_llm(load_local, lambda r: httpx.Response(404, json={"error": "model 'qwen3:4b-instruct' not found"}))
    with pytest.raises(LLMError, match="HTTP 404: model 'qwen3:4b-instruct' not found"):
        await collect(llm)
    broken = ndjson({"message": {"content": "par"}, "done": False}, {"error": "OOM"})
    llm, _ = make_llm(load_local, lambda r: httpx.Response(200, content=broken))
    with pytest.raises(LLMError, match="Ollama: OOM"):
        await collect(llm)


async def test_unreachable_server_is_llm_unavailable(load_local):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    llm, _ = make_llm(load_local, refuse)
    with pytest.raises(LLMUnavailableError, match=r"unreachable at http://127\.0\.0\.1:11434"):
        await collect(llm)

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    llm, _ = make_llm(load_local, slow)
    with pytest.raises(LLMUnavailableError, match="timed out"):
        await llm.generate(MESSAGES)


class Route(BaseModel):
    intent: Literal["document_qa", "general_qa"]
    needs_retrieval: bool
    query_en: str | None = None


async def test_generate_json_constrains_and_validates(load_local):
    reply = {"message": {"role": "assistant", "content": '{"intent": "document_qa", "needs_retrieval": true}'}}
    llm, rec = make_llm(load_local, lambda r: httpx.Response(200, json={**reply, "done": True}))
    route = await llm.generate_json(MESSAGES, Route)
    assert route == Route(intent="document_qa", needs_retrieval=True)
    body = rec.body()
    assert body["stream"] is False and body["format"] == Route.model_json_schema()
    assert body["options"]["temperature"] == 0.0  # router temperature
    assert body["model"] == "qwen3:4b-instruct"  # router model
    assert "num_predict" not in body["options"]
    await llm.generate_json(MESSAGES, Route, max_tokens=96)  # the router bounds its output
    assert rec.body()["options"]["num_predict"] == 96


async def test_generate_json_rejects_output_that_does_not_validate(load_local):
    reply = {"message": {"role": "assistant", "content": '{"intent": "chit_chat"}'}, "done": True}
    llm, _ = make_llm(load_local, lambda r: httpx.Response(200, json=reply))
    with pytest.raises(LLMError, match="doesn't match Route"):
        await llm.generate_json(MESSAGES, Route)


async def test_preload_loads_the_chat_model_with_the_answers_context_size(load_local):
    llm, rec = make_llm(load_local, lambda r: httpx.Response(200, json={"done": True, "done_reason": "load"}))
    await llm.preload()
    assert rec.requests[0].url.path == "/api/generate"
    # no prompt: Ollama only loads the model; same num_ctx as answers, or the first answer reloads it
    assert rec.body() == {"model": "qwen3:4b-instruct", "options": {"num_ctx": 8192}, "keep_alive": "30m"}


async def test_preload_failures_are_llm_errors(load_local):
    llm, _ = make_llm(load_local, lambda r: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(LLMError, match="model not found"):
        await llm.preload()
    down, _ = make_llm(load_local, lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused", request=r)))
    with pytest.raises(LLMUnavailableError, match="unreachable"):
        await down.preload()


# ------------------------------------------------------------------ one model load (B6)


async def test_every_request_keeps_the_same_context_size_and_keep_alive(load_local):
    """Ollama reloads the model when a request asks for another num_ctx: router, answers, titles, summaries, memory,
    warm-ups and preload all take it from one place."""

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        if body.get("format"):
            content = '{"intent": "document_qa", "needs_retrieval": true}'
            return httpx.Response(200, json={"message": {"role": "assistant", "content": content}, "done": True})
        return httpx.Response(200, content=chunks("ok"))

    llm, rec = make_llm(load_local, respond)
    await llm.preload()
    await collect(llm, max_tokens=384)  # an answer
    await llm.generate(MESSAGES, temperature=0.1, max_tokens=200)  # memory summary, title
    await llm.generate_json(MESSAGES, Route, max_tokens=96)  # router
    await llm.generate_json(MESSAGES, Route, model="qwen3:4b-instruct")  # chat summary
    await llm.warm_up([MESSAGES, MESSAGES])
    bodies = [rec.body(i) for i in range(len(rec.requests))]
    assert len(bodies) == 7
    assert {b["options"]["num_ctx"] for b in bodies} == {8192}
    assert {b["keep_alive"] for b in bodies} == {"30m"}
    assert [b["options"].get("num_predict") for b in bodies[-2:]] == [1, 1]  # warm-ups read, they don't write


async def test_warm_prompts_reads_the_router_and_answer_prompts_once(load_local):
    from app.services.preload import warm_prompts
    from app.services.prompts import answer_system_prompt
    from app.services.router import ROUTER_SYSTEM_PROMPT

    llm, rec = make_llm(load_local, lambda r: httpx.Response(200, content=chunks("ok")))
    await warm_prompts(llm, load_local())
    systems = [rec.body(i)["messages"][0]["content"] for i in range(len(rec.requests))]
    # the answer's system prompt is the same in English and Hindi (the question asks for the language): read once
    assert systems == [ROUTER_SYSTEM_PROMPT, answer_system_prompt("en", "short")]
    assert {rec.body(i)["options"]["num_predict"] for i in range(2)} == {1}  # read, not answered


def test_openai_compatible_placeholder_raises(cloud_settings):
    from app.providers.registry import build_container

    llm = build_container(cloud_settings, allow_placeholders=True)["llm"]
    with pytest.raises(NotImplementedError, match="placeholder"):
        llm.stream  # noqa: B018
