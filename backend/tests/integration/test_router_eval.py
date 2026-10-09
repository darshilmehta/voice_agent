"""The turn router on the real model (qwen3:4b-instruct via Ollama) over a labelled set of EN / HI / Hinglish turns.

    RUN_INTEGRATION=1 uv run pytest tests/integration/test_router_eval.py -s

Needs only Ollama (127.0.0.1:11434, the configured router model pulled); no embedder, reranker, speech models or
Qdrant. Prints, per intent, how often the router model (after application validation) got the intent right, how
often the whole router did (keyword fast path first, as in a turn), whether rewritten / English queries carry the
expected words, and the router call's latency (wall clock, plus Ollama's prompt and output token counts and times).

B1 cases (``documents_match``) also run the planner's check of "general" proposals for questions about facts against
a cached retrieval with that score, and count how many document questions reach the documents with and without it.
"""

from __future__ import annotations

import json
import statistics
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.domain.conversation import ConversationState
from app.domain.projects import Message
from app.providers.base import ProviderContext
from app.providers.llm import LLMMessage, OllamaLLM
from app.services.planning import TurnPlanner, interrupted_answer, live_search, policy
from app.services.prompts import answer_system_prompt
from app.services.retrieval import Confidence, RetrievalResult
from app.services.router import (
    RouteRequest,
    RouterProposal,
    fast_route,
    router_messages,
    validate,
    with_live_tools,
)
from app.settings import load_settings

from .conftest import LOCAL_CONFIG, METRICS

pytestmark = pytest.mark.integration

CASES = json.loads((Path(__file__).parent / "router_cases.json").read_text(encoding="utf-8"))
MIN_INTENT_ACCURACY = 0.85  # router model + validation, over every case
MIN_QUERY_ACCURACY = 0.8  # rewritten / English queries carrying the expected words


def _messages(history: list[list[str]]) -> list[Message]:
    out = []
    for i, item in enumerate(history, start=1):
        role, text = item[0], item[1]
        heard = item[2] if len(item) > 2 else None
        out.append(
            Message(
                id=f"msg_{i}",
                chat_id="cht_eval",
                seq=i,
                role=role,  # type: ignore[arg-type]
                modality="voice",
                text=text,
                heard_text=heard,
                language=None,
                citations=[],
                route={"stopped": True, "interrupted": "barge_in", "answer": "grounded"}
                if heard
                else {"answer": "grounded"},
                latency=None,
                created_at=datetime(2026, 10, 9, tzinfo=UTC),
            )
        )
    return out


def request_for(case: dict[str, Any]) -> RouteRequest:
    history = _messages(case.get("history", []))
    state = ConversationState(
        chat_id="cht_eval",
        active_topic=case.get("topic"),
        document_topic=case.get("document_topic") or case.get("topic"),
        last_intent="document_qa" if case.get("topic") else None,
    )
    return RouteRequest(
        case["utterance"],
        "en",
        history,
        state,
        CASES["documents"],
        interrupted_answer(history),
        available_tools=frozenset({"web_search"}),  # live-data cases (§3.7) check the tool decision too
    )


class CachedRetrieval:
    """A speculative retrieval whose best passage scores ``score`` (B1 cases)."""

    def __init__(self, score: float) -> None:
        self.score = score

    async def result_for(self, query: str, query_en: str | None):
        confidence = Confidence(self.score, self.score, 0.5, self.score >= 0.05)
        return RetrievalResult(query, query_en or query, [], 8, confidence), "used"

    def discard(self) -> None:
        pass


async def checked(decision, req: RouteRequest, score: float):
    """The decision after the planner's B1 check against a retrieval scoring ``score``, and the answer mode."""
    planner = TurnPlanner(None, None, timeout_s=2.5)  # type: ignore[arg-type]
    decision, prefetched = await planner._check_facts(decision, req, CachedRetrieval(score))  # type: ignore[arg-type]
    if prefetched is not None:
        await prefetched
    p = policy(decision, req, has_documents=True, retrieval_enabled=True)
    return p.decision, p.mode


def _contains_all(text: str | None, groups: list[list[str]]) -> bool:
    low = (text or "").casefold()
    return all(any(word.casefold() in low for word in group) for group in groups)


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


async def test_router_on_the_real_model(tmp_path):
    settings = load_settings(LOCAL_CONFIG, {"APP_ROOT_DIR": str(tmp_path)})
    async with httpx.AsyncClient() as http:
        try:
            tags = (await http.get(f"{settings.llm.base_url}/api/tags", timeout=2)).json()
        except httpx.HTTPError as e:
            pytest.skip(f"Ollama not reachable at {settings.llm.base_url}: {e}")
        if settings.llm.router_model not in {m["name"] for m in tags.get("models", [])}:
            pytest.skip(f"{settings.llm.router_model} not pulled")
        llm = OllamaLLM(settings.llm, ProviderContext(settings=settings, http=http))

        async def call(req: RouteRequest) -> tuple[RouterProposal, dict[str, Any], float]:
            """One router call as ``generate_json`` makes it, keeping Ollama's timings."""
            body = llm.request_body(
                router_messages(req),
                model=settings.llm.router_model,
                temperature=settings.llm.temperature.router,
                stream=False,
                max_tokens=96,
                format=RouterProposal.model_json_schema(),
            )
            t0 = time.perf_counter()
            r = await http.post(f"{llm.base_url}/api/chat", json=body, timeout=60)
            wall = (time.perf_counter() - t0) * 1000
            data = r.json()
            return RouterProposal.model_validate_json(data["message"]["content"]), data, wall

        await call(request_for(CASES["cases"][0]))  # warm-up: load the model with the configured context

        by_intent: dict[str, list[bool]] = defaultdict(list)
        pipeline_ok: list[bool] = []
        query_ok: list[bool] = []
        walls: list[float] = []
        prompt_tokens: list[int] = []
        prefill_ms: list[float] = []
        output_tokens: list[int] = []
        compute: list[float] = []
        misses: list[str] = []
        tools_ok: list[bool] = []
        web_ok: list[bool] = []
        web_queries: list[str] = []
        b1_router: list[bool] = []
        b1_checked: list[bool] = []
        fast = 0
        for case in CASES["cases"]:
            req = request_for(case)
            proposal, data, wall = await call(req)
            decision = validate(proposal, req, llm_ms=wall)
            route = decision.route
            accepted = {case["intent"], *case.get("also", [])}
            ok = route.intent in accepted
            by_intent[case["intent"]].append(ok)
            quick = fast_route(req)
            fast += quick is not None
            pipeline_ok.append((quick.route.intent if quick is not None else route.intent) in accepted)
            if "query" in case or "query_en" in case:
                q_ok = True
                if "query" in case:
                    q_ok &= _contains_all(route.rewritten_query or req.utterance, case["query"])
                if "query_en" in case:
                    q_ok &= _contains_all(route.query_en, case["query_en"])
                query_ok.append(q_ok)
                if not q_ok:
                    misses.append(f"{case['id']}: query={route.rewritten_query!r} query_en={route.query_en!r}")
            if not ok:
                misses.append(f"{case['id']}: {proposal.intent} → {route.intent}, expected {sorted(accepted)}")
            if "tools" in case:  # live data (§3.7): the tool decision and the query that would leave the machine
                live, web_q, _ = live_search(with_live_tools(quick or decision, req), req)
                tools_ok.append(list(live.route.tools) == case["tools"])
                if not tools_ok[-1]:
                    misses.append(f"{case['id']}: tools {list(live.route.tools)}, expected {case['tools']}")
                if "web_query" in case:
                    web_ok.append(_contains_all(web_q, case["web_query"]))
                    web_queries.append(f"{case['id']}: {web_q!r}")
                    if not web_ok[-1]:
                        misses.append(f"{case['id']}: web query {web_q!r}")
            if "documents_match" in case:  # B1: does the turn reach the documents (or stay general, as expected)?
                first = quick or decision
                final, mode = await checked(first, req, case["documents_match"])
                want_search = case.get("final") != "general"
                b1_router.append((first.route.intent in ("document_qa", "mixed", "correction")) == want_search)
                b1_checked.append((mode in ("grounded", "mixed")) == want_search)
                if not b1_checked[-1]:
                    misses.append(f"{case['id']}: after the B1 check {final.route.intent} / {mode}")
            walls.append(wall)
            prompt_tokens.append(data.get("prompt_eval_count", 0))
            prefill_ms.append(data.get("prompt_eval_duration", 0) / 1e6)
            output_tokens.append(data.get("eval_count", 0))
            compute.append((data.get("prompt_eval_duration", 0) + data.get("eval_duration", 0)) / 1e6)

    total = sum(len(v) for v in by_intent.values())
    correct = sum(sum(v) for v in by_intent.values())
    for intent, results in sorted(by_intent.items()):
        METRICS[f"router intent {intent}"] = f"{sum(results)}/{len(results)}"
    METRICS["router intent accuracy (model + validation)"] = f"{correct}/{total} = {correct / total:.0%}"
    METRICS["router intent accuracy (fast path first)"] = (
        f"{sum(pipeline_ok)}/{len(pipeline_ok)}; fast path took {fast}/{len(pipeline_ok)} turns"
    )
    METRICS["router queries (rewritten / English)"] = f"{sum(query_ok)}/{len(query_ok)}"
    METRICS["router live-data tools (web_search or none)"] = f"{sum(tools_ok)}/{len(tools_ok)}"
    METRICS["router web search queries carrying the expected words"] = f"{sum(web_ok)}/{len(web_ok)}"
    METRICS["router web search queries"] = "\n  " + "\n  ".join(web_queries)
    METRICS["B1 cases answered as they should (search the documents, or stay general): router alone"] = (
        f"{sum(b1_router)}/{len(b1_router)}"
    )
    METRICS["B1 cases answered as they should: with the planner's check of general proposals"] = (
        f"{sum(b1_checked)}/{len(b1_checked)}"
    )
    METRICS["router call latency p50 / p95 / max (wall)"] = (
        f"{statistics.median(walls):.0f} / {_pct(walls, 0.95):.0f} / {max(walls):.0f} ms"
    )
    METRICS["router model time p50 / p95 (Ollama prefill + generation)"] = (
        f"{statistics.median(compute):.0f} / {_pct(compute, 0.95):.0f} ms"
    )
    METRICS["router prompt tokens p50 (prefill ms p50)"] = (
        f"{statistics.median(prompt_tokens):.0f} ({statistics.median(prefill_ms):.0f} ms)"
    )
    METRICS["router output tokens p50 / max"] = f"{statistics.median(output_tokens):.0f} / {max(output_tokens)}"
    if misses:
        METRICS["router misses"] = "\n  " + "\n  ".join(misses)
    assert correct / total >= MIN_INTENT_ACCURACY, misses
    assert sum(query_ok) / len(query_ok) >= MIN_QUERY_ACCURACY, misses
    assert all(tools_ok), misses  # application code decides, from the utterance and the model's English query
    assert all(b1_checked), misses
    assert sum(web_ok) / len(web_ok) >= MIN_QUERY_ACCURACY, misses


async def test_what_a_router_call_costs_the_next_answer(tmp_path):
    """Ollama reuses the KV cache of a prompt's prefix from the previous request (one slot unless OLLAMA_NUM_PARALLEL
    > 1). Consecutive answers share their system prompt and history; a router call between them evicts that prefix,
    so the answer's prefill grows. Reports the answer's prefill with and without a router call in between."""
    settings = load_settings(LOCAL_CONFIG, {"APP_ROOT_DIR": str(tmp_path)})
    async with httpx.AsyncClient() as http:
        try:
            await http.get(f"{settings.llm.base_url}/api/tags", timeout=2)
        except httpx.HTTPError as e:
            pytest.skip(f"Ollama not reachable at {settings.llm.base_url}: {e}")
        llm = OllamaLLM(settings.llm, ProviderContext(settings=settings, http=http))
        passage = "EBITDA margin improved to 18.2% in FY24 from 16.9% in FY23, on enterprise revenue growth. " * 40
        system = answer_system_prompt("en", "short")
        history = [LLMMessage("user", "What was revenue in FY24?"), LLMMessage("assistant", "It was ₹4,210 crore.")]

        async def answer(question: str, nonce: str = "") -> float:
            """The answer prompt's prefill time (ms); a ``nonce`` at its start defeats any cached prefix."""
            messages = [
                LLMMessage("system", nonce + system),
                *history,
                LLMMessage("user", f"Sources:\n{passage}\n{question}"),
            ]
            body = llm.request_body(
                messages, model=settings.llm.chat_model, temperature=0.0, stream=False, max_tokens=1
            )
            data = (await http.post(f"{llm.base_url}/api/chat", json=body, timeout=60)).json()
            return data["prompt_eval_duration"] / 1e6

        async def route() -> None:
            body = llm.request_body(
                router_messages(request_for(CASES["cases"][4])),
                model=settings.llm.router_model,
                temperature=0.0,
                stream=False,
                max_tokens=96,
                format=RouterProposal.model_json_schema(),
            )
            await http.post(f"{llm.base_url}/api/chat", json=body, timeout=60)

        cold, cached, after_router = [], [], []
        for i in range(3):
            cold.append(await answer("What was the EBITDA margin in FY24?", nonce=f"[{uuid.uuid4().hex[:8]}] "))
            cached.append(await answer(f"And in FY2{i}?"))  # same prefix as the previous request
            await route()
            after_router.append(await answer(f"And in FY1{i}?"))  # a router call came in between

    METRICS["answer prefill (~1.6k tokens) cold / prefix cached / after a router call"] = (
        f"{statistics.median(cold):.0f} / {statistics.median(cached):.0f} / {statistics.median(after_router):.0f} ms"
    )
