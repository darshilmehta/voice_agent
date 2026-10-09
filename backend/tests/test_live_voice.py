"""Live data over the transports (docs/DESIGN.md §3.7): ``tool`` events over SSE and the voice WebSocket, the
pre-synthesized filler spoken first, the answer's chunks after it (and how long after it), barge-in during the
search cancelling it, and the filler synthesized at startup."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from app.providers.llm import LLMMessage
from app.providers.registry import build_container
from app.services.preload import ModelPreloader
from app.services.voice.fillers import FILLERS, filler_audio
from app.services.voice.protocol import parse_frame
from app.services.voice.speech_text import SpokenChunk, heard_text
from app.settings import load_settings

from .conftest import LOCAL_CONFIG, mock_http
from .fakes import FakeTTS, FakeWebSearch, web_result
from .test_chat_api import ask, names, new_chat, payload, project_with_report
from .test_voice_api import Voice, VoiceClient, kinds, of, one

LIVE_Q = "Revenue grew 34% per the report; how is the stock doing today?"
LIVE, CORRECTION = 70, 40  # tones (what the user "says")
FILLER = FILLERS["en"]


def live_reply(messages: list[LLMMessage]) -> str:
    system = messages[0].content
    if system.startswith("You keep the memory"):
        return "- memory"
    if "can look up live data on the web" in system:
        return "Up 2% today [W1]. The report says revenue grew 34% [S1]."
    return "The EBITDA margin was 18.2% [S1]."


@pytest.fixture
def live(make_app, fakes) -> Iterator[Voice]:
    fakes.web = FakeWebSearch([(0.0, web_result(1)), (0.0, web_result(2))])
    fakes.llm.reply = live_reply
    fakes.stt.scripts.update({LIVE: LIVE_Q, CORRECTION: "No wait, what was the EBITDA margin in FY24?"})
    with make_app(TOOLS__WEB_SEARCH__TIMEOUT_S="2", TOOLS__WEB_SEARCH__PARTIAL_WAIT_MS="50") as api:
        project, _ = project_with_report(api)
        yield Voice(api, fakes, new_chat(api, project))


CONTINUED = "A second source adds heavy volume [W2]."
CONTINUED_SPOKEN = "A second source adds heavy volume."


def continuing_reply(messages: list[LLMMessage]) -> str:
    """As ``live_reply``; the prompt for results that arrived after the answer gets a sentence citing the new one."""
    if "More web results arrived" in messages[-1].content:
        return CONTINUED
    return live_reply(messages)


@pytest.fixture
def continuing(make_app, fakes) -> Iterator[Voice]:
    """Two results, the second 0.3 s after the answer started; two continuations are allowed and the search then
    stays open for the rest of its 4 s, so the turn is still waiting for it after the first continuation."""
    fakes.web = FakeWebSearch([(0.0, web_result(1)), (0.3, web_result(2))])
    fakes.web.hang = True
    fakes.llm.reply = continuing_reply
    fakes.stt.scripts.update({LIVE: LIVE_Q, CORRECTION: "No wait, what was the EBITDA margin in FY24?"})
    env = {
        "TOOLS__WEB_SEARCH__TIMEOUT_S": "4",
        "TOOLS__WEB_SEARCH__PARTIAL_WAIT_MS": "50",
        "TOOLS__WEB_SEARCH__MAX_CONTINUATIONS": "2",
    }
    with make_app(**env) as api:
        project, _ = project_with_report(api)
        yield Voice(api, fakes, new_chat(api, project))


class TimedClient(VoiceClient):
    """Also records when each message arrived."""

    def __init__(self, ws: Any) -> None:
        self.arrivals: list[tuple[float, Any]] = []
        self._lock = threading.Lock()
        super().__init__(ws)

    def _read(self) -> None:
        while True:
            try:
                message = self.ws.receive()
            except Exception:
                return
            if message["type"] == "websocket.close":
                return
            item = json.loads(message["text"]) if message.get("text") is not None else message["bytes"]
            with self._lock:
                self.arrivals.append((time.perf_counter(), item))
            self.inbox.put(item)

    def arrived(self, check) -> float:
        with self._lock:
            return next(t for t, item in self.arrivals if check(item))


# ------------------------------------------------------------------ SSE


def test_tool_events_over_sse(live):
    events = ask(live.api, live.chat, LIVE_Q)
    assert names(events) == [
        "user_message", "tool", "tool", "tool", "sources", *["delta"] * (len(events) - 6), "agent_message",
    ]  # fmt: skip
    tools = [d for e, d in events if e == "tool"]
    assert tools[0] == {"name": "web_search", "phase": "start", "query": "how is the stock doing today?"}
    assert tools[1]["phase"] == "results" and [s["source_id"] for s in tools[1]["sources"]] == ["W1", "W2"]
    assert tools[2]["phase"] == "done" and tools[2]["count"] == 2
    sources = payload(events, "sources")["sources"]
    assert [s.get("kind", "document") for s in sources][-2:] == ["web", "web"]
    agent = payload(events, "agent_message")
    assert [c.get("kind", "document") for c in agent["citations"]] == ["web", "document"]
    assert agent["citations"][0]["url"] == "https://news.example.com/story-1"
    # the transcript shows the same, and the web citation keeps its link
    saved = live.transcript()[-1]
    assert saved["citations"] == agent["citations"] and saved["route"]["tools"] == ["web_search"]


# ------------------------------------------------------------------ voice


def test_the_filler_is_spoken_at_once_then_the_answer(live):
    with live.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(LIVE)
        items = c.until("agent_message")
    main = [m for m in items if not (isinstance(m, dict) and m["type"] == "transcript_partial")]
    assert kinds(main)[:8] == [
        "user_speech", "user_speech", "state", "user_message", "turn", "tool", "audio_chunk", "frame",
    ]  # fmt: skip
    # F12: the filler (announce and all its frames) is sent before the turn's next message, always
    filler_at = main.index(of(items, "audio_chunk")[0])
    assert all(isinstance(m, bytes) for m in main[filler_at + 1 : filler_at + 4])
    assert main.index(one(items, "sources")) > filler_at + 3
    # F13: "speaking" comes with the answer's first audio, not with the filler
    assert main.index({"type": "state", "state": "speaking"}) > main.index(of(items, "audio_chunk")[1])
    tool_events = of(items, "tool")
    assert [t["phase"] for t in tool_events] == ["start", "results", "done"]
    assert all(t["turn_id"] == 1 for t in tool_events)
    chunks = of(items, "audio_chunk")
    assert chunks[0] == {
        "type": "audio_chunk", "turn_id": 1, "chunk_index": 0, "text": FILLER, "duration_ms": 500.0, "filler": True,
    }  # fmt: skip
    assert [ch["text"] for ch in chunks[1:]] == ["Up 2% today.", "The report says revenue grew 34%."]
    assert all("filler" not in ch for ch in chunks[1:])
    assert main.index(one(items, "sources")) < main.index(of(items, "delta")[0])  # sources still before deltas
    assert kinds(main)[-1] == "agent_message"
    agent = one(items, "agent_message")["message"]
    assert FILLER not in agent["text"] and agent["heard_text"] is None  # the filler isn't part of the answer
    # pre-synthesized: synthesized once (on first use here: preload only runs with web search enabled)
    assert live.fakes.tts.calls[0] == (FILLER, "en")


def test_the_first_words_follow_the_filler_within_a_second(live):
    """A search whose first results arrive 300 ms in: the answer's first audio follows the filler's end by < 1 s."""
    live.fakes.web.script = [(0.3, web_result(1)), (0.0, web_result(2))]
    with live.connect() as ws:
        c = TimedClient(ws)
        c.start()
        c.say(LIVE)
        items = c.until("agent_message")
        filler_at = c.arrived(lambda m: isinstance(m, dict) and m.get("filler") is True)
        first_at = c.arrived(lambda m: isinstance(m, dict) and m.get("type") == "audio_chunk" and m["chunk_index"] == 1)
    filler_ms = of(items, "audio_chunk")[0]["duration_ms"]
    gap_ms = (first_at - filler_at) * 1000 - filler_ms
    assert gap_ms < 1000, gap_ms  # first words < 1 s after the filler (here they are ready before it ends)


def test_barge_in_during_the_search_cancels_it(live):
    live.fakes.web.script, live.fakes.web.hang = [], True
    with live.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(LIVE)
        c.until(lambda m: isinstance(m, dict) and m.get("filler") is True)
        # the filler alone doesn't make the agent "speaking" (F13): the answer hasn't started
        assert not [m for m in c.quiet(0.2) if isinstance(m, dict) and m.get("state") == "speaking"]
        assert live.fakes.web.open == 1  # searching
        live.fakes.web.hang = False
        c.send("barge_in_start", turn_id=1, played_ms=600)  # the filler has played
        c.say(CORRECTION, ms=400)
        items = c.until("agent_message")
        assert one(items, "barge_in") == {"type": "barge_in", "turn_id": 1, "decision": "stop"}
        stopped = one(items, "agent_message")["message"]
        assert stopped["heard_text"] == "" and stopped["route"]["interrupted"] == "barge_in"
        assert stopped["route"]["web_search"]["status"] == "cancelled" or stopped["route"]["web_search"] is None
        assert not [m for m in items if isinstance(m, dict) and m.get("type") == "tool"]  # nothing after the cut
        deadline = time.monotonic() + 2
        while live.fakes.web.open and time.monotonic() < deadline:
            time.sleep(0.01)
        assert live.fakes.web.cancelled == 1 and live.fakes.web.open == 0
        nxt = c.until("agent_message")  # the correction is answered (no live cue: no search, no filler)
        assert one(nxt, "turn")["turn_id"] == 2 and not of(nxt, "tool")
        assert not [ch for ch in of(nxt, "audio_chunk") if ch.get("filler")]


LAST_SENTENCE = "The report says revenue grew 34%."


def test_speech_after_a_heard_answer_while_the_web_still_runs_is_not_a_barge_in(live):
    """F10: the answer is complete and was played; the turn only waits for slower engines (a continuation may
    come). The user speaking now ends the turn: the answer is saved complete (not stopped, no heard_text), no
    barge_in decision, no "interrupted" state, and the speech is the next turn. The answer's last sentence was spoken
    without waiting for the search to end."""
    live.fakes.web.script, live.fakes.web.hang = [(0.0, web_result(1))], True  # one engine never answers
    with live.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(LIVE)
        items = c.until(lambda m: isinstance(m, dict) and m.get("text") == LAST_SENTENCE)
        assert not [t for t in of(items, "tool") if t["phase"] in ("done", "timeout")]  # the search still runs
        items += c.until(lambda m: isinstance(m, bytes) and parse_frame(m)[1:3] == (2, 2))  # its last frame (600 ms)
        c.send("playback", turn_id=1, played_ms=60_000)  # the client played all of it
        c.say(CORRECTION, ms=400)
        after = c.until("turn")
    assert not of(after, "barge_in") and {"type": "state", "state": "interrupted"} not in after
    agent = one(after, "agent_message")["message"]
    assert agent["text"] == "Up 2% today [W1]. The report says revenue grew 34% [S1]."
    assert agent["heard_text"] is None and agent["route"]["stopped"] is False and "interrupted" not in agent["route"]
    assert after.index(one(after, "agent_message")) < after.index(one(after, "user_message"))
    assert one(after, "turn")["turn_id"] == 2
    assert live.fakes.web.cancelled == 1


def test_a_kept_continuation_is_spoken_at_once_not_when_the_turn_ends(continuing):
    """N2: a continuation sentence goes to the speech chunker's flush as soon as it is kept, not with the turn's final
    flush. The turn here keeps waiting for the search (a further continuation) for seconds after it, so an unspoken
    sentence would sit in the chunker all that time, and speech after the answer's played audio would count as
    "answer fully heard" with the continuation neither spoken nor part of what was heard."""
    with continuing.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(LIVE)
        items = c.until(lambda m: isinstance(m, dict) and m.get("text") == CONTINUED_SPOKEN, timeout=2.5)
        # (unfixed, that chunk comes with the search's end at 4 s)
        assert not [t for t in of(items, "tool") if t["phase"] in ("done", "timeout")]  # the search is still open
        assert not of(items, "agent_message")
        deltas = of(items, "delta")
        assert deltas[-1]["text"] == " " + CONTINUED
        assert items.index(deltas[-1]) < items.index(of(items, "audio_chunk")[-1])  # shown before it is spoken
        assert [ch["text"] for ch in of(items, "audio_chunk")][-3:] == [
            "Up 2% today.", "The report says revenue grew 34%.", CONTINUED_SPOKEN,
        ]  # fmt: skip
        # all of it played: the next speech is no barge-in, and the saved answer has the continuation
        items += c.until(lambda m: isinstance(m, bytes) and parse_frame(m)[1:3] == (3, 2))  # its last frame
        c.send("playback", turn_id=1, played_ms=60_000)
        c.say(CORRECTION, ms=400)
        after = c.until("turn")
    assert not of(after, "barge_in") and {"type": "state", "state": "interrupted"} not in after
    agent = one(after, "agent_message")["message"]
    assert agent["text"] == "Up 2% today [W1]. The report says revenue grew 34% [S1]. " + CONTINUED
    assert agent["heard_text"] is None and agent["route"]["stopped"] is False and "interrupted" not in agent["route"]


def test_a_tts_failure_on_the_filler_doesnt_silence_the_answer(live, monkeypatch):
    original = FakeTTS.synthesize

    async def synthesize(self, text, language):
        if text == FILLER:
            raise RuntimeError("voice not loaded")
        return await original(self, text, language)

    monkeypatch.setattr(FakeTTS, "synthesize", synthesize)
    with live.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(LIVE)
        items = c.until("agent_message")
    assert not of(items, "error")
    assert [ch["text"] for ch in of(items, "audio_chunk")] == ["Up 2% today.", "The report says revenue grew 34%."]


def test_heard_text_skips_the_filler():
    chunks = [
        SpokenChunk(0, FILLER, 0, 500, filler=True),
        SpokenChunk(1, "Today the stock is up 2%.", 500, 600),
    ]
    assert heard_text(chunks, 300) == ""  # cut during the filler: nothing of the answer was heard
    assert heard_text(chunks, 800) == "Today the stock"
    assert heard_text(chunks, 2000) == "Today the stock is up 2%."


async def test_the_filler_is_synthesized_at_startup_when_web_search_is_on(write_config, tmp_path, fakes):
    allowed = write_config(LOCAL_CONFIG, strict_offline_exceptions=["web_search"])
    settings = load_settings(allowed, {"APP_ROOT_DIR": str(tmp_path), "TOOLS__WEB_SEARCH__ENABLED": "true"})
    container = fakes.install(build_container(settings, http=mock_http()))
    preloader = ModelPreloader(container)
    preloader.start()
    await preloader.wait()
    assert filler_audio(fakes.tts).cached("en") is not None and filler_audio(fakes.tts).cached("hi") is not None
    assert fakes.tts.calls == [(FILLERS["en"], "en"), (FILLERS["hi"], "hi")]


async def test_the_filler_is_not_synthesized_when_web_search_is_off(load_local, fakes):
    container = fakes.install(build_container(load_local(), http=mock_http()))
    preloader = ModelPreloader(container)
    preloader.start()
    await preloader.wait()
    assert fakes.tts.calls == []
