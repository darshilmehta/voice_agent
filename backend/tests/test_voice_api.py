"""WS /ws/chats/{id}/voice end to end with fake VAD, STT, TTS, retrieval and LLM (no ML, Qdrant or Ollama).

Audio encodes what is "said" in its level (see ``fakes.speech``): FakeVAD hears speech above 0.02 and FakeSTT maps
the level to a script. FakeTTS makes 100 ms of audio per word, so what was heard at a given ``played_ms`` is exact.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.providers.llm import LLMUnavailableError
from app.services.messages import MessageService
from app.services.prompts import ABSTENTIONS
from app.services.voice.protocol import parse_frame

from .conftest import Fakes
from .fakes import OUT_RATE, silence, speech
from .test_chat_api import EN, HI, new_chat, project_with_report

REPLY = "The EBITDA margin was 18.2% [S1]. Revenue grew 34% in FY24 [S2]. The board approved a dividend."
SPOKEN = ["The EBITDA margin was 18.2%.", "Revenue grew 34% in FY24.", "The board approved a dividend."]
HOLD_AFTER_TWO_SENTENCES = 11  # 6-character pieces: "… FY24 [S2]. T" completes the second chunk
QUESTION, CORRECTION, MMHMM, HINDI = 30, 40, 50, 60  # tones (what the user "says")


@dataclass(frozen=True)
class Closed:
    code: int | None


Item = dict[str, Any] | bytes | Closed


class VoiceClient:
    """The test side of a voice session: a reader thread collects every frame, in order."""

    def __init__(self, ws: Any) -> None:
        self.ws = ws
        self.inbox: queue.Queue[Item] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        while True:
            try:
                message = self.ws.receive()
            except Exception:
                self.inbox.put(Closed(None))
                return
            if message["type"] == "websocket.close":
                self.inbox.put(Closed(message.get("code")))
                return
            if message.get("text") is not None:
                self.inbox.put(json.loads(message["text"]))
            elif message.get("bytes") is not None:
                self.inbox.put(message["bytes"])

    def next(self, timeout: float = 5.0) -> Item:
        return self.inbox.get(timeout=timeout)

    def until(self, match: str | Callable[[Item], bool], timeout: float = 5.0) -> list[Item]:
        """Every message up to and including the first that matches (a message type, or a predicate)."""
        check = match if callable(match) else (lambda m: isinstance(m, dict) and m["type"] == match)
        got: list[Item] = []
        deadline = time.monotonic() + timeout
        while True:
            item = self.next(timeout=max(0.01, deadline - time.monotonic()))
            got.append(item)
            if check(item):
                return got

    def quiet(self, seconds: float = 0.3) -> list[Item]:
        """Whatever arrives within ``seconds``."""
        got: list[Item] = []
        deadline = time.monotonic() + seconds
        while (left := deadline - time.monotonic()) > 0:
            try:
                got.append(self.inbox.get(timeout=left))
            except queue.Empty:
                break
        return got

    def send(self, type_: str, **fields: Any) -> None:
        self.ws.send_text(json.dumps({"type": type_, **fields}))

    def audio(self, pcm: bytes, frame_ms: int = 32) -> None:
        size = 16 * frame_ms * 2
        for i in range(0, len(pcm), size):
            self.ws.send_bytes(pcm[i : i + size])

    def say(self, tone: int, ms: int = 600, then_silence_ms: int = 700) -> None:
        self.audio(speech(ms, tone) + silence(then_silence_ms))

    def start(self, language: str | None = "en") -> dict[str, Any]:
        self.send("start", language=language)
        ready = self.next()
        assert isinstance(ready, dict) and ready["type"] == "ready", ready
        assert self.next() == {"type": "state", "state": "listening"}
        return ready


def kinds(items: list[Item]) -> list[str]:
    """Message types in order, runs of frames and deltas collapsed; transcript_partial (speculative, its timing
    relative to other messages isn't fixed) left out."""
    out = []
    for m in items:
        kind = "frame" if isinstance(m, bytes) else "closed" if isinstance(m, Closed) else m["type"]
        if kind == "transcript_partial":
            continue
        if not (out and kind in ("frame", "delta") and out[-1] == kind):
            out.append(kind)
    return out


def of_turn(items: list[Item], turn_id: int) -> list[Item]:
    """Deltas, audio chunks and frames of one turn."""
    return [
        m
        for m in items
        if (isinstance(m, bytes) and parse_frame(m)[0] == turn_id)
        or (isinstance(m, dict) and m["type"] in ("delta", "audio_chunk") and m["turn_id"] == turn_id)
    ]


def of(items: list[Item], type_: str) -> list[dict[str, Any]]:
    return [m for m in items if isinstance(m, dict) and m["type"] == type_]


def one(items: list[Item], type_: str) -> dict[str, Any]:
    (found,) = of(items, type_)
    return found


def chunk_index(i: int) -> Callable[[Item], bool]:
    """Matches the last frame of audio chunk ``i`` (FakeTTS chunks of 5 words: 500 ms = frames 0, 1, 2)."""
    return lambda m: isinstance(m, bytes) and parse_frame(m)[1:3] == (i, 2)


@dataclass
class Voice:
    api: TestClient
    fakes: Fakes
    chat: str

    def connect(self, chat: str | None = None) -> Any:
        return self.api.websocket_connect(f"/ws/chats/{chat or self.chat}/voice")

    def transcript(self) -> list[dict[str, Any]]:
        return self.api.get(f"/api/chats/{self.chat}/messages").json()["items"]


@pytest.fixture
def voice(make_app, fakes) -> Iterator[Voice]:
    fakes.llm.reply = REPLY
    fakes.stt.scripts.update(
        {
            QUESTION: EN,
            CORRECTION: "No wait, I meant the EBITDA margin",
            MMHMM: "Mm-hmm.",
            HINDI: HI,
        }
    )
    with make_app() as api:
        project, _ = project_with_report(api)
        yield Voice(api, fakes, new_chat(api, project))


# ------------------------------------------------------------------ connecting


def test_ready_and_state_after_start(voice):
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.send("start", language="hi")
        ready = c.next()
        assert ready["session_id"].startswith("vsn_")
        assert ready == {
            "type": "ready",
            "session_id": ready["session_id"],
            "input_sample_rate": 16_000,
            "output_sample_rate": 24_000,
            "language": "hi",
        }
        assert c.next() == {"type": "state", "state": "listening"}


def test_unknown_chat_is_closed_with_4404(voice):
    with voice.connect("cht_unknown") as ws:
        assert VoiceClient(ws).next() == Closed(4404)


def test_a_second_session_for_the_chat_closes_the_first_with_4409(voice):
    with voice.connect() as ws1:
        first = VoiceClient(ws1)
        first.start()
        with voice.connect() as ws2:
            second = VoiceClient(ws2)
            assert first.until(lambda m: isinstance(m, Closed))[-1] == Closed(4409)
            second.start()
            second.say(QUESTION)
            assert one(second.until("agent_message"), "user_message")["message"]["text"] == EN


def test_audio_before_start_and_malformed_input_are_errors_but_the_session_stays(voice):
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.audio(silence(64))
        assert c.next() == {
            "type": "error",
            "detail": 'audio before start is ignored: send {"type": "start"} first',
            "stage": "audio",
        }
        c.start()
        for _ in range(5):
            ws.send_bytes(b"\x00\x01\x02")  # odd length: not PCM16
        ws.send_text("not json")
        c.send("dance")
        errors = c.quiet(0.5)
        assert [e["detail"] for e in of(errors, "error")] == [  # one error per kind, not one per bad message
            "audio frames must be PCM16 mono 16 kHz, at most 1 s (got 3 bytes)",
            "control messages must be JSON: Expecting value",
        ]
        assert {e["stage"] for e in of(errors, "error")} == {"audio"}
        c.say(QUESTION)
        assert one(c.until("agent_message"), "agent_message")["message"]["role"] == "agent"


# ------------------------------------------------------------------ a spoken turn


def test_a_spoken_question_is_answered_in_order_with_audio(voice):
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        items = c.until("agent_message")
        main = [m for m in items if not (isinstance(m, dict) and m["type"] == "transcript_partial")]
        assert kinds(main)[:7] == ["user_speech", "user_speech", "state", "user_message", "turn", "sources", "delta"]
        assert [m["phase"] for m in of(items, "user_speech")] == ["start", "end"]
        assert of(items, "state") == [{"type": "state", "state": "thinking"}, {"type": "state", "state": "speaking"}]
        assert of(items, "transcript_partial") == [{"type": "transcript_partial", "text": EN}]  # speculative STT
        assert one(items, "turn") == {"type": "turn", "turn_id": 1}
        assert kinds(main)[-1] == "agent_message"

        # every chunk is announced, then its frames follow in order (other messages may come in between), headed
        # (turn, chunk, seq), before the next chunk is announced
        chunks = of(items, "audio_chunk")
        assert [ch["text"] for ch in chunks] == SPOKEN
        assert [(ch["turn_id"], ch["chunk_index"], ch["duration_ms"]) for ch in chunks] == [
            (1, 0, 500.0),
            (1, 1, 500.0),
            (1, 2, 500.0),
        ]
        for ch, following in zip(chunks, [*chunks[1:], None], strict=True):
            at, until = main.index(ch), (main.index(following) if following else len(main))
            frames = [parse_frame(f) for f in main[at + 1 : until] if isinstance(f, bytes)]
            assert [(t, i, s) for t, i, s, _ in frames] == [(1, ch["chunk_index"], s) for s in range(3)]
            assert sum(len(pcm) for *_, pcm in frames) == ch["duration_ms"] * OUT_RATE // 1000 * 2
        assert main.index(chunks[0]) > main.index(of(items, "delta")[0])
        assert main.index({"type": "state", "state": "speaking"}) > main.index(chunks[0])
        assert "".join(d["text"] for d in of(items, "delta")) == REPLY
        assert all(d["turn_id"] == 1 for d in of(items, "delta"))

        user, agent = one(items, "user_message")["message"], one(items, "agent_message")["message"]
        assert (user["role"], user["text"], user["modality"], user["language"]) == ("user", EN, "voice", "en")
        assert user["latency"]["speculative_stt"] is True and user["latency"]["speech_ms"] >= 576
        assert (agent["modality"], agent["heard_text"], agent["interrupted"]) == ("voice", None, False)
        assert agent["text"] == REPLY and [s["source_id"] for s in agent["citations"]] == ["S1", "S2"]
        assert agent["route"]["length"] == "short" and agent["route"]["stopped"] is False
        sources = one(items, "sources")
        assert set(sources) == {"type", "sources", "confidence", "abstained"} and sources["abstained"] is False
        assert voice.transcript() == [user, agent]  # everything said is in the chat's transcript

        c.send("playback_done", turn_id=1)
        assert c.next() == {"type": "state", "state": "listening"}

    assert len(voice.fakes.stt.calls) == 1 and voice.fakes.stt.calls[0]["languages"] == ["en"]  # speculative reused
    assert voice.fakes.tts.calls == [(text, "en") for text in SPOKEN]
    assert voice.fakes.llm.calls[0]["max_tokens"] == 384  # the short, speakable answer style


def test_speech_resuming_after_a_pause_discards_the_speculative_transcript(voice):
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.audio(speech(500, QUESTION) + silence(400) + speech(500, QUESTION) + silence(700))
        assert one(c.until("user_message"), "user_message")["message"]["latency"]["speculative_stt"] is True
    assert len(voice.fakes.stt.calls) == 2  # paused once, resumed, paused again: the first result was not used


def test_short_noises_are_ignored(voice):
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION, ms=150)  # under min_speech_ms (250)
        # The noise gate (§3.10) announces a turn only once it has min_speech_ms of speech: nothing reaches the client
        assert of(c.quiet(0.5), "user_speech") == []
    assert voice.fakes.stt.calls == [] and voice.transcript() == []


def test_short_noises_are_announced_and_closed_without_the_noise_gate(make_app, fakes):
    fakes.stt.scripts[QUESTION] = EN
    with make_app(VOICE__NOISE__ADAPTIVE_GATING="false") as api:
        project, _ = project_with_report(api)
        chat = new_chat(api, project)
        with api.websocket_connect(f"/ws/chats/{chat}/voice") as ws:
            c = VoiceClient(ws)
            c.start()
            c.say(QUESTION, ms=150)
            assert [m["phase"] for m in of(c.quiet(0.5), "user_speech")] == ["start", "end"]
    assert fakes.stt.calls == []


def test_a_chat_without_ready_documents_still_talks_and_abstains(make_app, fakes):
    fakes.stt.scripts[QUESTION] = EN
    with make_app() as api:
        project = api.post("/api/projects", json={"name": "Empty"}).json()["id"]
        chat = new_chat(api, project)
        with api.websocket_connect(f"/ws/chats/{chat}/voice") as ws:
            c = VoiceClient(ws)
            c.start()
            c.say(QUESTION)
            items = c.until("agent_message")
    assert one(items, "sources")["abstained"] is True
    assert one(items, "agent_message")["message"]["text"] == ABSTENTIONS["no_documents"]["en"]
    assert " ".join(ch["text"] for ch in of(items, "audio_chunk")) == ABSTENTIONS["no_documents"]["en"]
    assert fakes.llm.calls == []


def test_spoken_language_is_detected_per_utterance(voice):
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start(language=None)
        voice.fakes.reranker.scorer = lambda q, passage: 0.9 if "18.2%" in passage else 0.05
        voice.fakes.llm.reply = "वित्त वर्ष 2024 में EBITDA मार्जिन 18.2% था [S1]।"
        c.say(HINDI)
        items = c.until("agent_message")
    user, agent = one(items, "user_message")["message"], one(items, "agent_message")["message"]
    assert (user["language"], agent["language"]) == ("hi", "hi")
    assert voice.fakes.stt.calls[0]["languages"] == ["en", "hi"]  # detection restricted to the configured languages
    assert voice.fakes.tts.calls == [("वित्त वर्ष 2024 में EBITDA", "hi"), ("मार्जिन 18.2% था।", "hi")]
    assert "Answer in Hindi" in voice.fakes.llm.calls[0]["messages"][-1].content


# ------------------------------------------------------------------ barge-in and stop


def test_barge_in_mid_answer_stops_it_saves_what_was_heard_and_answers_the_correction(voice):
    voice.fakes.llm.hold_after = HOLD_AFTER_TWO_SENTENCES  # the model is still generating the third sentence
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until(chunk_index(1))
        voice.fakes.llm.hold_after = None  # the next answer streams normally
        c.send("barge_in_start", turn_id=1, played_ms=700)  # 2 chunks of 500 ms sent; 40% into the second
        c.say(CORRECTION, ms=400)
        items = c.until("agent_message")
        # Speech start, the decision, "interrupted", then the cut answer. The correction may end (user_speech end,
        # state thinking) before the cut answer is saved on a slow machine: §3.10 only promises the cut answer's
        # agent_message comes before the next turn's user_message.
        assert kinds(items)[:3] == ["user_speech", "barge_in", "state"] and kinds(items)[-1] == "agent_message"
        assert set(kinds(items)[3:-1]) <= {"user_speech", "state"}
        assert one(items, "barge_in") == {"type": "barge_in", "turn_id": 1, "decision": "stop"}
        assert of(items, "state")[0] == {"type": "state", "state": "interrupted"}
        stopped = one(items, "agent_message")["message"]
        assert stopped["heard_text"] == "The EBITDA margin was 18.2%. Revenue grew"
        assert stopped["interrupted"] is True
        assert stopped["text"].startswith("The EBITDA margin was 18.2% [S1]. Revenue grew 34% in FY24 [S2].")
        assert stopped["route"]["stopped"] is True and stopped["route"]["interrupted"] == "barge_in"
        assert voice.fakes.llm.closed  # generation was cancelled

        # the interruption becomes the next turn; nothing of turn 1 comes after the decision
        before = items
        items = c.until("agent_message")
        seen = kinds(before) + kinds(items)
        # The correction's end of speech and "thinking" come after the decision and before its user_message,
        # which comes before its turn id; nothing of turn 1 follows the cut answer.
        assert seen.index("barge_in") < len(seen) - 1 - seen[::-1].index("user_speech") < seen.index("user_message")
        assert seen.index("user_message") < seen.index("turn")
        assert {"type": "state", "state": "thinking"} in of(before, "state") + of(items, "state")
        assert one(items, "user_message")["message"]["text"] == "No wait, I meant the EBITDA margin"
        assert one(items, "turn")["turn_id"] == 2  # turn ids increase within the session
        assert not of_turn(items, 1) and of_turn(items, 2)
        c.send("playback_done", turn_id=2)
        c.until(lambda m: m == {"type": "state", "state": "listening"})

    roles = [(m["role"], m["text"][:12], m["heard_text"]) for m in voice.transcript()]
    assert roles == [
        ("user", EN[:12], None),
        ("agent", "The EBITDA m", "The EBITDA margin was 18.2%. Revenue grew"),
        ("user", "No wait, I m", None),
        ("agent", "The EBITDA m", None),
    ]
    # the next answer's prompt carries what was heard, not the whole interrupted answer
    history = [(m.role, m.content) for m in voice.fakes.llm.calls[1]["messages"][1:-1]]
    assert history == [("user", EN), ("assistant", "The EBITDA margin was 18.2%. Revenue grew")]


def test_barge_in_after_the_answer_was_generated_records_what_was_heard(voice):
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        complete = one(c.until("agent_message"), "agent_message")["message"]
        c.send("barge_in_start", turn_id=1, played_ms=1250)
        c.say(CORRECTION, ms=400)
        items = c.until("agent_message")
        assert one(items, "barge_in")["decision"] == "stop"
        updated = one(items, "agent_message")["message"]
    assert updated["id"] == complete["id"] and updated["text"] == REPLY
    assert updated["heard_text"] == "The EBITDA margin was 18.2%. Revenue grew 34% in FY24. The board"
    assert updated["route"]["stopped"] is True and updated["route"]["interrupted"] == "barge_in"
    assert voice.transcript()[1]["heard_text"] == updated["heard_text"]


def test_a_backchannel_lets_the_answer_go_on(voice):
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until("agent_message")
        c.send("barge_in_start", turn_id=1, played_ms=300)
        c.say(MMHMM, ms=320)
        items = c.until("barge_in")
        assert items[-1] == {"type": "barge_in", "turn_id": 1, "decision": "resume"}
        assert not of(c.quiet(0.3), "user_message")  # "mm-hmm" is not a turn
        c.send("playback_done", turn_id=1)
        assert c.until("state")[-1] == {"type": "state", "state": "listening"}
    assert [m["role"] for m in voice.transcript()] == ["user", "agent"]
    assert voice.transcript()[1]["heard_text"] is None


def test_an_utterance_while_thinking_cancels_the_answer_even_without_barge_in_start(voice):
    voice.fakes.llm.hold_after = 0  # nothing generated yet
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until("sources")
        voice.fakes.llm.hold_after = None
        c.say(CORRECTION)
        items = c.until(lambda m: isinstance(m, dict) and m["type"] == "user_message")
        assert kinds(items) == [
            "user_speech",
            "user_speech",
            "barge_in",
            "state",  # interrupted
            "agent_message",  # turn 1, closed before anything of turn 2
            "state",  # thinking: the correction is being answered
            "user_message",
        ]
        assert one(items, "barge_in") == {"type": "barge_in", "turn_id": 1, "decision": "stop"}
        stopped = one(items, "agent_message")["message"]
        assert (stopped["text"], stopped["heard_text"], stopped["route"]["stopped"]) == ("", "", True)
        assert one(c.until("agent_message"), "turn")["turn_id"] == 2
        c.send("playback_done", turn_id=2)
        c.until(lambda m: m == {"type": "state", "state": "listening"})
    assert [(m["role"], m["text"]) for m in voice.transcript()] == [
        ("user", EN),
        ("agent", ""),  # nothing was generated: an empty, stopped answer closes the turn
        ("user", "No wait, I meant the EBITDA margin"),
        ("agent", REPLY),
    ]


def test_nothing_of_a_cut_turn_is_sent_after_the_decision(voice):
    voice.fakes.tts.delay = 0.2  # the second chunk is still being synthesized when the decision is made
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until(chunk_index(0))
        c.send("barge_in_start", turn_id=1, played_ms=200)
        c.say(CORRECTION, ms=400, then_silence_ms=0)
        before = c.until("barge_in")
        after = c.until("agent_message") + c.quiet(0.6)  # long enough for the pending synthesis to finish
    assert before[-1]["decision"] == "stop"
    assert not of_turn(after, 1)  # no delta, audio_chunk or frame of turn 1 after barge_in
    assert one(after, "agent_message")["message"]["heard_text"] == "The EBITDA"


def test_stop_before_any_answer_text_closes_the_turn_with_an_empty_answer(voice):
    voice.fakes.llm.hold_after = 0  # thinking: no delta yet
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until("sources")
        c.send("stop")
        items = c.until(lambda m: m == {"type": "state", "state": "listening"})
    assert kinds(items) == ["state", "agent_message", "state"]
    stopped = one(items, "agent_message")["message"]
    assert (stopped["text"], stopped["heard_text"], stopped["citations"]) == ("", "", [])
    assert stopped["route"]["stopped"] is True and stopped["route"]["interrupted"] == "stop"
    assert [m["role"] for m in voice.transcript()] == ["user", "agent"]


def test_stale_barge_in_is_answered_with_resume(voice):
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.send("barge_in_start", turn_id=99, played_ms=0)
        assert c.next() == {"type": "barge_in", "turn_id": 99, "decision": "resume"}


def test_stop_cancels_the_answer_and_keeps_what_was_heard(voice):
    voice.fakes.llm.hold_after = HOLD_AFTER_TWO_SENTENCES
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until(chunk_index(1))
        c.send("playback", turn_id=1, played_ms=600)
        c.send("stop")
        items = c.until(lambda m: m == {"type": "state", "state": "listening"})
        assert not of_turn(c.quiet(0.2), 1)
    assert kinds(items) == ["state", "agent_message", "state"]
    stopped = one(items, "agent_message")["message"]
    assert stopped["heard_text"] == "The EBITDA margin was 18.2%. Revenue"
    assert stopped["route"]["stopped"] is True and stopped["route"]["interrupted"] == "stop"


def test_ending_the_session_mid_answer_saves_it_and_closes_normally(voice):
    voice.fakes.llm.hold_after = HOLD_AFTER_TWO_SENTENCES
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until(chunk_index(1))
        c.send("playback", turn_id=1, played_ms=1000)
        c.send("end")
        assert c.until(lambda m: isinstance(m, Closed))[-1] == Closed(1000)
    agent = voice.transcript()[-1]
    assert agent["heard_text"] == "The EBITDA margin was 18.2%. Revenue grew 34% in FY24."
    assert agent["route"]["interrupted"] == "disconnect" and agent["route"]["stopped"] is True


# ------------------------------------------------------------------ failures keep the session open


def _fail_stt(v: Voice, mp: pytest.MonkeyPatch) -> None:
    v.fakes.stt.fail_with = RuntimeError("model crashed")


def _fail_retrieval(v: Voice, mp: pytest.MonkeyPatch) -> None:
    v.fakes.store.fail_with = ConnectionError("Qdrant unreachable")


def _fail_llm(v: Voice, mp: pytest.MonkeyPatch) -> None:
    v.fakes.llm.fail_with = LLMUnavailableError("Ollama unreachable")


def _fail_tts(v: Voice, mp: pytest.MonkeyPatch) -> None:
    v.fakes.tts.fail_with = RuntimeError("no voice")


def _fail_storage(v: Voice, mp: pytest.MonkeyPatch) -> None:
    real = MessageService.append

    async def append(self, chat_id, *, role, **kw):
        if role == "agent":
            raise RuntimeError("database is locked")
        return await real(self, chat_id, role=role, **kw)

    mp.setattr(MessageService, "append", append)


def _heal(v: Voice, mp: pytest.MonkeyPatch) -> None:
    v.fakes.stt.fail_with = v.fakes.llm.fail_with = v.fakes.tts.fail_with = v.fakes.store.fail_with = None
    mp.undo()


@pytest.mark.parametrize(
    ("break_", "stage", "detail"),
    [
        (_fail_stt, "stt", "transcription failed: RuntimeError: model crashed"),
        (_fail_retrieval, "retrieval", "document search failed: ConnectionError: Qdrant unreachable"),
        (_fail_llm, "llm", "answer generation failed: LLMUnavailableError: Ollama unreachable"),
        (_fail_tts, "tts", "speech synthesis failed: RuntimeError: no voice"),
        (_fail_storage, "storage", "could not save the answer: RuntimeError: database is locked"),
    ],
)
def test_a_failing_stage_is_reported_and_the_next_turn_works(voice, monkeypatch, break_, stage, detail):
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        break_(voice, monkeypatch)
        c.say(QUESTION)
        items = c.until("error")
        assert items[-1] == {"type": "error", "detail": detail, "stage": stage}
        if stage == "tts":  # the answer is still shown, only once reported, and the turn ends
            rest = c.until(lambda m: m == {"type": "state", "state": "listening"})
            assert kinds(rest) in (["delta", "agent_message", "state"], ["agent_message", "state"])
            assert not any(isinstance(m, bytes) for m in rest)
        else:
            c.until(lambda m: m == {"type": "state", "state": "listening"})
        _heal(voice, monkeypatch)
        c.say(QUESTION)
        items = c.until("agent_message")
        assert of(items, "audio_chunk") and not of(items, "error")


def test_tts_failing_partway_reports_once_then_closes_the_turn_after_the_audio_sent(voice):
    voice.fakes.tts.fail_with, voice.fakes.tts.fail_from = RuntimeError("no voice"), 1  # the second chunk fails
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        items = c.until("agent_message")
        last_frame = max(i for i, m in enumerate(items) if isinstance(m, bytes))
        error_at = items.index(one(items, "error"))
        assert last_frame < error_at < len(items) - 1  # first chunk's audio, then the error, then agent_message last
        assert one(items, "error")["stage"] == "tts" and len(of(items, "audio_chunk")) == 1
        assert one(items, "agent_message")["message"]["text"] == REPLY  # the full answer is still saved and shown
        c.send("playback_done", turn_id=1)
        assert c.next() == {"type": "state", "state": "listening"}


# ------------------------------------------------------------------ startup


def test_b5_the_voice_follows_the_script_of_the_answer(voice):
    """Asked in English, answered in Hindi even after one more try: saved as Hindi and spoken by the Hindi voice,
    the whole answer (its "EBITDA 18.2%" chunk too), not by the English voice reading Devanagari."""
    voice.fakes.llm.reply = "FY24 में EBITDA मार्जिन 18.2% था [S1]। EBITDA 18.2% [S2]। राजस्व 34% बढ़ा।"
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start(language="en")
        c.say(QUESTION)
        agent = one(c.until("agent_message"), "agent_message")["message"]
    assert agent["language"] == "hi" and agent["route"]["language"] == "en"
    spoken = [lang for _, lang in voice.fakes.tts.calls]
    assert len(spoken) >= 2 and set(spoken) == {"hi"}


def test_health_reports_the_model_preload(voice):
    body = voice.api.get("/health").json()
    preload = body["preload"]
    assert preload["state"] == "ready"
    assert [m["capability"] for m in preload["models"]] == ["vad", "stt", "tts", "embeddings", "reranker", "llm"]
    assert all(m["state"] == "ready" and m["seconds"] is not None for m in preload["models"])
