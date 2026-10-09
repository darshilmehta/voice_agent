"""Instant visuals on the real models (docs/DESIGN.md §12.1, "Instant draft, then refine"; §10 phase 10: a cited chart
on screen ≤ 1 s after the spoken answer starts), opt-in:

    CANVAS_PARSE_CACHE=<folder> RUN_INTEGRATION=1 uv run --group ml pytest \
        tests/integration/test_canvas_instant_visual.py -s

The real voice session over the WebSocket, with the real router, answer and planner (Ollama, qwen3:4b-instruct) and
the real speech synthesis (Kokoro), over the typed tables of the Valmora annual report. What the user says comes from
scripted speech (the fake VAD and STT of the unit tests: tones stand for questions, so no Whisper), and retrieval is
cached as in ``test_answer_latency.py`` (the corpus's passages; the pages a question is about rank first, after the
measured idle-machine search and rerank times), so no embedder, reranker or Qdrant.

Per question, timed where the client receives it: the answer's first audio, the visual's draft (``visual ready``),
the planner's refined visual (a second ``ready`` in place) or its withdrawal, ``agent_message``, and whether the
spoken tail ("It's on screen now.") was said; the route's ``visual_plan`` (confident draft, planner skipped / same /
changed). Then the first-token check: the same question answered with the canvas (the draft is built while the
answer is written) and without it, alternately, on fresh chats: the answer's first delta from the turn's start.
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.offline import apply_runtime_env
from app.providers.registry import build_container
from app.services.chat_turns import AgentMessageEvent, ChatTurnService, DeltaEvent, wait_for_background
from app.services.chats import ChatService
from app.services.projects import ProjectService
from app.settings import load_settings

from ..fakes import FakeSTT, FakeVAD, silence, speech
from .canvas_corpus import CorpusDocument, corpus_fixture  # noqa: F401  (the "corpus" fixture)
from .conftest import LOCAL_CONFIG, METRICS, require_chat_model
from .test_answer_latency import REPORT, CachedIndex, CachedStore
from .test_canvas_conversation_e2e import load_documents
from .test_voice_e2e import PRELOAD_TIMEOUT_S, LiveClient, Received, is_type

pytestmark = pytest.mark.integration

# (tone, what is said, the report's pages it is about)
QUESTIONS = [
    (31, "Show me Valmora's revenue by quarter for FY24", (19,)),
    (32, "How did each segment's revenue change from FY23 to FY24?", (18,)),
    (33, "Show me the key financial highlights", (3,)),
    (34, "Show me the shareholding pattern", (28,)),
    (35, "How has the workforce changed between FY23 and FY24? Show me.", (12,)),
    (36, "Which segment has the highest EBITDA margin?", (18,)),
    (37, "Walk me through the cash flow for FY24", (25,)),
    (38, "Show me Valmora's revenue", (3, 18, 19)),
    (39, "वालमोरा की FY24 की तिमाही आय का चार्ट दिखाओ", (19,)),
    (40, "FY24 aur FY23 ke quarterly EBITDA ki tulna dikhao", (19, 20)),
]
FIRST_TOKEN_QUESTION = ("Show me Valmora's revenue by quarter for FY24", (19,))
FIRST_TOKEN_RUNS = int(os.environ.get("CANVAS_FIRST_TOKEN_RUNS", "6"))
TURN_TIMEOUT_S = 60
ROUTE_KEYS = ("intent", "answer", "visual", "abstained")
REFINE_WAIT_S = 14  # the planner's timeout and building it, with room


def pcm(tone: int) -> bytes:
    return speech(900, tone) + silence(1200)


@pytest.fixture
def app(corpus: dict[str, CorpusDocument], models_root: Path, tmp_path: Path):
    env = {
        **os.environ,
        "APP_ROOT_DIR": str(tmp_path),
        "MODEL_CACHE__HF_HOME": str(models_root / "huggingface"),
        "INGESTION__ARTIFACTS_PATH": str(models_root / "docling"),
    }
    settings = load_settings(LOCAL_CONFIG, env)
    apply_runtime_env(settings)
    require_chat_model(settings)
    container = build_container(settings)
    index = CachedIndex()
    store = CachedStore(index)
    stt = FakeSTT()
    stt.scripts.update({tone: text for tone, text, _ in QUESTIONS})
    container.providers.update(embeddings=index, reranker=index, vector_store=store, vad=FakeVAD(), stt=stt)
    with TestClient(create_app(settings, container)) as client:
        t0 = time.perf_counter()
        while (report := client.get("/health").json()["preload"])["state"] == "loading":
            assert time.perf_counter() - t0 < PRELOAD_TIMEOUT_S, report
            time.sleep(0.5)
        assert report["state"] == "ready", report
        yield client, container, index, store


def settled_plan(api: TestClient, chat_id: str, message_id: str) -> dict[str, Any]:
    """The answer's ``route.visual_plan`` once its visual has settled (the planner may still be refining the draft
    when the answer has played)."""
    deadline = time.monotonic() + REFINE_WAIT_S
    while True:
        items = api.get(f"/api/chats/{chat_id}/messages?limit=100").json()["items"]
        route = next((m["route"] or {} for m in items if m["id"] == message_id), {})
        plan = route.get("visual_plan") or {}
        if plan.get("planner") is not None or not route.get("visual") or route.get("visual") == "none":
            return plan
        if route.get("visual_status") not in (None, "ready") or time.monotonic() > deadline:
            return plan
        time.sleep(0.2)


def received(log: list[Received], since: float, check) -> list[Received]:
    return [r for r in log if r.at >= since and check(r.item)]


def setup_chat(
    api: TestClient, container: Any, store: CachedStore, corpus: dict[str, CorpusDocument]
) -> tuple[str, str]:
    canvas = api.app.state.canvas
    db = container["metadata_db"]

    async def setup() -> tuple[str, str]:
        project = await ProjectService(db).create("Instant visuals")
        ids = await load_documents(db, project.id, corpus, store, canvas)
        chat = await ChatService(db).create(project.id, document_scope=[ids[REPORT]])
        return project.id, chat.id

    return api.portal.call(setup)


def voice_rows(api: TestClient, chat_id: str, index: CachedIndex, questions: list[tuple[int, str, tuple[int, ...]]]):
    """Each question said in one voice session, timed where the client receives it."""
    rows: list[dict[str, Any]] = []
    with api.websocket_connect(f"/ws/chats/{chat_id}/voice") as ws:
        c = LiveClient(ws)
        c.send("start", language=None)
        c.wait(is_type("state", state="listening"))
        for tone, text, pages in questions:
            index.pages = pages
            since = time.perf_counter()
            for i in range(0, len(audio := pcm(tone)), 1024):
                ws.send_bytes(audio[i : i + 1024])
            user = c.wait(is_type("user_message"), since=since, timeout=TURN_TIMEOUT_S)
            agent = c.wait(is_type("agent_message"), since=since, timeout=TURN_TIMEOUT_S)
            c.wait(is_type("state", state="listening"), since=agent.at, timeout=TURN_TIMEOUT_S)  # played out
            plan = settled_plan(api, chat_id, agent.item["message"]["id"])
            time.sleep(0.3)  # a refined visual that came with the settling
            log = received(c.drain(), since, lambda m: True)

            def ms(a: float | None, b: float | None) -> int | None:
                return None if a is None or b is None else round((b - a) * 1000)

            visuals = [r for r in log if isinstance(r.item, dict) and r.item.get("type") == "visual"]
            readies = [r for r in visuals if r.item["phase"] == "ready"]
            chunks = [r for r in log if isinstance(r.item, dict) and r.item.get("type") == "audio_chunk"]
            first_audio = next((r.at for r in chunks if not r.item.get("filler") and not r.item.get("tail")), None)
            deltas = [r.at for r in log if isinstance(r.item, dict) and r.item.get("type") == "delta"]
            rows.append(
                {
                    "text": text,
                    "kinds": [r.item["visual"]["kind"] for r in readies],
                    "failed": [r.item.get("detail") for r in visuals if r.item["phase"] == "failed"],
                    "first_audio_ms": ms(user.at, first_audio),
                    "draft_after_first_audio_ms": ms(first_audio, readies[0].at) if readies else None,
                    "visual_before_first_audio": bool(visuals)
                    and first_audio is not None
                    and visuals[0].at < first_audio,
                    "refined_after_first_audio_ms": ms(first_audio, readies[1].at) if len(readies) > 1 else None,
                    "refined_after_answer_text_ms": ms(deltas[-1] if deltas else None, readies[1].at)
                    if len(readies) > 1
                    else None,
                    "tail": any(r.item.get("tail") for r in chunks),
                    "plan": plan,
                    "route": {k: agent.item["message"]["route"].get(k) for k in ROUTE_KEYS},
                    "said": agent.item["message"]["text"][:100],
                }
            )
    return rows


def report(prefix: str, rows: list[dict[str, Any]]) -> None:
    for r in rows:
        METRICS[f"{prefix}: {r['text'][:52]!r}"] = (
            f"visual {r['kinds'] or 'none'}{' failed ' + str(r['failed']) if r['failed'] else ''}; first audio "
            f"{r['first_audio_ms']} ms; draft +{r['draft_after_first_audio_ms']} ms after it; refined "
            f"+{r['refined_after_first_audio_ms']} ms (+{r['refined_after_answer_text_ms']} after the text); "
            f"tail {r['tail']}; plan {r['plan']}; route {r['route']}; said {r['said']!r}"
        )
    shown = [r for r in rows if r["draft_after_first_audio_ms"] is not None]
    METRICS[f"{prefix}: first audio → draft on screen (ms, each)"] = [r["draft_after_first_audio_ms"] for r in shown]
    refined = [r for r in rows if r["refined_after_first_audio_ms"] is not None]
    METRICS[f"{prefix}: first audio → refined visual (ms, each)"] = [r["refined_after_first_audio_ms"] for r in refined]
    METRICS[f"{prefix}: answer's text → refined visual (ms, each)"] = [
        r["refined_after_answer_text_ms"] for r in refined
    ]
    METRICS[f"{prefix}: tail spoken"] = f"{sum(r['tail'] for r in rows)}/{len(shown)} visuals"
    outcomes = [str(r["plan"].get("planner")) for r in rows if r["plan"]]
    METRICS[f"{prefix}: planner outcomes"] = {o: outcomes.count(o) for o in sorted(set(outcomes))}
    planner_ms = [r["plan"]["planner_ms"] for r in rows if r["plan"].get("planner_ms") is not None]
    if planner_ms:
        METRICS[f"{prefix}: planner call (ms, each)"] = planner_ms


def test_instant_visuals_in_a_voice_session(app, corpus):
    api, container, index, store = app
    project_id, chat_id = setup_chat(api, container, store, corpus)
    rows = voice_rows(api, chat_id, index, QUESTIONS)
    first_token = first_token_ab(api, container, project_id, index)
    report("voice", rows)
    METRICS["first delta from the turn's start, draft beside the answer / visual after it (ms, each)"] = first_token
    if all(first_token.values()):
        METRICS["first delta p50, draft beside the answer / visual after it"] = (
            f"{statistics.median(first_token['draft']):.0f} / {statistics.median(first_token['after']):.0f} ms"
        )
    assert not any(r["visual_before_first_audio"] for r in rows)  # §3.10: never before the first audio
    assert any(r["draft_after_first_audio_ms"] is not None for r in rows), rows


def test_refined_visuals_in_a_voice_session(app, corpus, monkeypatch):
    """The refine path timed: every draft taken as unsure (as the ones the planner has to check), so the planner
    runs after each answer and its chart replaces the draft in place, or confirms it."""
    import app.services.canvas.service as service

    real = service.draft_visual

    def unsure(*args: Any, **kwargs: Any) -> Any:
        d = real(*args, **kwargs)
        d.confident = False
        return d

    monkeypatch.setattr(service, "draft_visual", unsure)
    api, container, index, store = app
    _, chat_id = setup_chat(api, container, store, corpus)
    rows = voice_rows(api, chat_id, index, [QUESTIONS[i] for i in (0, 1, 3, 7, 8)])
    report("voice, every draft refined", rows)
    assert not any(r["visual_before_first_audio"] for r in rows)


def first_token_ab(api: TestClient, container: Any, project_id: str, index: CachedIndex) -> dict[str, list[int]]:
    """The answer's first delta from the turn's start with the draft built beside the answer ("draft": as turns do
    now) and with the visual started only once the answer's text is complete ("after": as before the draft; the
    answer's prompt is the same), alternately, each on a fresh chat."""
    text, pages = FIRST_TOKEN_QUESTION
    out: dict[str, list[int]] = {"draft": [], "after": []}

    async def one(draft: bool) -> int:
        db = container["metadata_db"]
        chat = await ChatService(db).create(project_id)
        service = ChatTurnService.from_container(container, canvas=api.app.state.canvas)
        if not draft:
            service._start_draft = _no_draft  # type: ignore[method-assign]
        index.pages = pages
        turn = await service.begin(chat.id, text, modality="voice", length="short")
        t0 = time.perf_counter()
        first = None
        async for event in service.run(turn, on_visual=_ignore):
            if isinstance(event, DeltaEvent) and first is None:
                first = time.perf_counter() - t0
            if isinstance(event, AgentMessageEvent):
                break
        await wait_for_background()
        return round((first or 0) * 1000)

    for n in range(FIRST_TOKEN_RUNS * 2):
        draft = n % 2 == 0
        out["draft" if draft else "after"].append(api.portal.call(one, draft))
        time.sleep(0.5)
    return out


async def _no_draft(*_args: Any) -> None:
    return None


async def _ignore(_event: Any) -> None:
    return None
