"""One model load for every kind of LLM call, and what warming the router's prompt saves (docs/DESIGN.md §9.5, B6).

    RUN_INTEGRATION=1 uv run pytest tests/integration/test_model_load.py -s

Needs only Ollama (127.0.0.1:11434, qwen3:4b-instruct pulled). Runs the app's calls through ``OllamaLLM`` in a mixed
order (startup preload and warm-up, router, answer, title, memory summary, chat summary, live-prompt warm-up) and
checks with ``/api/ps`` and each reply's ``load_duration`` that the model was never reloaded: same context size,
loaded once. Then measures a router call whose prompt Ollama has not read before (as for the first routed turn after
startup without the warm-up) against one whose prompt the warm-up has read.
"""

from __future__ import annotations

import statistics
import uuid

import httpx
import pytest

from app.domain.conversation import ConversationState
from app.providers.base import ProviderContext
from app.providers.llm import LLMMessage, OllamaLLM
from app.services import router as router_module
from app.services.preload import warm_prompts
from app.services.prompts import answer_system_prompt, memory_system_prompt
from app.services.revisit_prompts import SummaryDraft, summary_system_prompt, title_messages
from app.services.router import LLMTurnRouter, RouteRequest
from app.settings import load_settings

from .conftest import LOCAL_CONFIG, METRICS
from .test_answer_latency import OllamaStats

pytestmark = pytest.mark.integration

RELOAD_MS = 500  # a reply that spent longer than this loading the model reloaded it


async def test_mixed_calls_never_reload_the_model(tmp_path, monkeypatch):
    settings = load_settings(LOCAL_CONFIG, {"APP_ROOT_DIR": str(tmp_path)})
    stats = OllamaStats()
    async with httpx.AsyncClient(transport=stats) as http:
        try:
            await http.get(f"{settings.llm.base_url}/api/tags", timeout=2)
        except httpx.HTTPError as e:
            pytest.skip(f"Ollama not reachable at {settings.llm.base_url}: {e}")
        llm = OllamaLLM(settings.llm, ProviderContext(settings=settings, http=http))
        await llm.preload()
        await warm_prompts(llm, settings)
        start = len(stats.done)
        req = RouteRequest("What was the EBITDA margin in FY24?", "en", [], ConversationState(chat_id="c"), ["r.pdf"])
        system = LLMMessage("system", answer_system_prompt("en", "short"))
        for i in range(2):
            await LLMTurnRouter(llm).propose(req)  # router
            question = LLMMessage("user", f"Sources: [S1] margin 18.2%.\nQuestion {i}?")
            await llm.generate([system, question], max_tokens=40)
            await llm.generate(title_messages("en", "What was the EBITDA margin?", "It was 18.2%."), max_tokens=24)
            exchange = LLMMessage("user", "User: margin?\nAssistant: 18.2%")
            memory = [LLMMessage("system", memory_system_prompt()), exchange]
            await llm.generate(memory, temperature=0.1, max_tokens=60)
            summary = [LLMMessage("system", summary_system_prompt("en")), LLMMessage("user", "#1 User: margin?")]
            await llm.generate_json(summary, SummaryDraft, model=settings.llm.chat_model, max_tokens=200)
            await llm.warm_up([[system, LLMMessage("user", "Document sources: [S1] margin 18.2%.")]])
        ps = (await http.get(f"{settings.llm.base_url}/api/ps", timeout=5)).json()

        # The router's prompt never read before (a nonce in front of it) against the warmed one, five times each.
        cold, warm = [], []
        prompt = router_module.ROUTER_SYSTEM_PROMPT
        for _ in range(5):
            monkeypatch.setattr(router_module, "ROUTER_SYSTEM_PROMPT", f"[{uuid.uuid4().hex[:8]}] {prompt}")
            await LLMTurnRouter(llm).propose(req)
            cold.append(stats.done[-1])
            monkeypatch.setattr(router_module, "ROUTER_SYSTEM_PROMPT", prompt)
            await LLMTurnRouter(llm).propose(req)
            warm.append(stats.done[-1])

    calls = stats.done[start:]
    loads = [round(s.get("load_duration", 0) / 1e6) for s in calls]
    model = next(m for m in ps["models"] if m["name"] == settings.llm.chat_model)
    METRICS["calls after startup (router, answer, title, memory, summary, warm-up) x2"] = len(calls)
    METRICS["load_duration of each call (ms)"] = loads
    METRICS["/api/ps context_length"] = model.get("context_length")

    def ms(s: dict) -> float:
        return (s.get("prompt_eval_duration", 0) + s.get("eval_duration", 0)) / 1e6

    METRICS["router call, prompt not read before / warmed (model time p50)"] = (
        f"{statistics.median(ms(s) for s in cold):.0f} / {statistics.median(ms(s) for s in warm):.0f} ms; prompt "
        f"reading {statistics.median(s['prompt_eval_duration'] / 1e6 for s in cold):.0f} / "
        f"{statistics.median(s['prompt_eval_duration'] / 1e6 for s in warm):.0f} ms"
    )
    assert model.get("context_length") in (None, settings.llm.num_ctx)
    assert max(loads) < RELOAD_MS, loads
