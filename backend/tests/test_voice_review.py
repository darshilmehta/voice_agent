"""Regression tests for the review of the voice session (PR #21) and the first real end-to-end run.

Each test names the finding it covers. Fakes only (see test_voice_api.py for the conventions: audio level encodes what
was said, 100 ms of TTS audio per word).
"""

# ruff: noqa: F811  (the `voice` fixture is imported from test_voice_api and requested by name)

from __future__ import annotations

import asyncio
import os
import stat
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.api import voice as voice_api
from app.providers import speech
from app.providers.llm import LLMError
from app.services.chat_turns import ChatTurnService
from app.services.messages import MessageService
from app.services.voice import session as session_module
from app.services.voice.protocol import MAX_MESSAGE_BYTES, parse_frame

from .fakes import silence
from .fakes import speech as speech_audio
from .test_voice_api import (  # noqa: F401  (voice is a fixture)
    CORRECTION,
    EN,
    HOLD_AFTER_TWO_SENTENCES,
    MMHMM,
    QUESTION,
    REPLY,
    Closed,
    VoiceClient,
    chunk_index,
    kinds,
    of,
    of_turn,
    one,
    voice,
)

HUM = 70  # a tone whose transcript is set per test


def is_state(state: str):
    return lambda m: m == {"type": "state", "state": state}


def roles(voice) -> list[tuple]:
    return [(m["role"], m["text"], m["heard_text"], (m["route"] or {}).get("interrupted")) for m in voice.transcript()]


# ------------------------------------------------------------------ bugs


def test_1_stop_while_the_utterance_is_transcribed_saves_it_unanswered(voice):
    voice.fakes.stt.delay = 0.6  # STT still running after the end of turn ("thinking" already shown)
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until(is_state("thinking"))
        c.send("stop")
        items = c.until(is_state("listening"), timeout=5)
        assert kinds(items) == ["state", "user_message", "turn", "agent_message", "state"]
        assert items[0] == {"type": "state", "state": "interrupted"}
        stopped = one(items, "agent_message")["message"]
        assert (stopped["text"], stopped["heard_text"], stopped["route"]["stopped"]) == ("", "", True)
        assert stopped["route"]["interrupted"] == "stop"
        assert not of(c.quiet(0.4), "audio_chunk")  # nothing is spoken
    assert roles(voice) == [("user", EN, None, None), ("agent", "", "", "stop")]
    assert voice.fakes.llm.calls == []


def test_2_stop_while_the_user_message_is_saved_keeps_the_turn(voice, monkeypatch):
    original = MessageService.append

    async def slow_user_save(self, chat_id, **kw):
        if kw.get("role") == "user":
            await asyncio.sleep(0.4)  # e.g. SQLite busy with an ingestion write
        return await original(self, chat_id, **kw)

    monkeypatch.setattr(MessageService, "append", slow_user_save)
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until(is_state("thinking"))
        time.sleep(0.1)  # the user message is being saved now
        c.send("stop")
        items = c.until(is_state("listening"), timeout=5)
        assert kinds(items) == ["state", "user_message", "turn", "agent_message", "state"]
        assert not of(c.quiet(0.4), "audio_chunk")
    assert roles(voice) == [("user", EN, None, None), ("agent", "", "", "stop")]


def test_3_stop_while_the_complete_answer_is_saved_keeps_it_with_what_was_heard(voice, monkeypatch):
    original = MessageService.append

    async def slow_answer_save(self, chat_id, **kw):
        if kw.get("role") == "agent" and not (kw.get("route") or {}).get("stopped"):
            await asyncio.sleep(0.5)  # a slow commit of the complete answer
        return await original(self, chat_id, **kw)

    monkeypatch.setattr(MessageService, "append", slow_answer_save)
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until(chunk_index(1))  # generation is done; its save is still running
        c.send("playback", turn_id=1, played_ms=600)
        c.send("stop")
        stopped = one(c.until("agent_message", timeout=5), "agent_message")["message"]
    assert stopped["text"] == REPLY and stopped["heard_text"] == "The EBITDA margin was 18.2%. Revenue"
    assert stopped["route"]["stopped"] is True and stopped["route"]["interrupted"] == "stop"
    assert roles(voice) == [("user", EN, None, None), ("agent", REPLY, stopped["heard_text"], "stop")]


# ------------------------------------------------------------------ risks


def test_4_a_session_dropped_mid_interrupt_does_not_leak_its_turn(voice, monkeypatch):
    original = session_module.VoiceSession._set_state

    async def slow_state(self, state):
        if state == "interrupted":
            await asyncio.sleep(0.5)  # e.g. waiting for a slow socket
        await original(self, state)

    monkeypatch.setattr(session_module.VoiceSession, "_set_state", slow_state)
    voice.fakes.llm.delay = 0.08  # still generating when the user cuts in
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until("sources")
        c.say(CORRECTION)
        c.until("barge_in")
        # the client drops now, while the interruption is still being settled
    time.sleep(1.5)
    assert voice.fakes.llm.closed and voice.fakes.llm.sent < 16  # generation stopped, didn't run to the end
    answers = [r for r in roles(voice) if r[0] == "agent"]
    assert answers and all(r[3] in ("barge_in", "disconnect") for r in answers)  # never saved as a complete answer


def test_5_an_unexpected_failure_inside_the_turn_frees_the_session(voice, monkeypatch):
    def broken_prompt(self, *a, **k):
        raise RuntimeError("bug in prompt building")

    monkeypatch.setattr(ChatTurnService, "_prompt", broken_prompt)
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        items = c.until(is_state("listening"))
        assert one(items, "error") == {
            "type": "error",
            "detail": "the answer failed: RuntimeError: bug in prompt building",
            "stage": "llm",
        }
        monkeypatch.undo()
        c.say(QUESTION)
        items = c.until("agent_message")
        assert not of(items, "barge_in") and of(items, "audio_chunk")  # not treated as interrupting a dead turn


def test_6_a_failure_handling_an_utterance_keeps_the_session_open(voice, monkeypatch):
    calls = {"n": 0}
    original = ChatTurnService.begin

    async def flaky_begin(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("database is locked")
        return await original(self, *a, **k)

    monkeypatch.setattr(ChatTurnService, "begin", flaky_begin)
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        items = c.until(is_state("listening"))
        assert one(items, "error")["stage"] == "storage" and "database is locked" in one(items, "error")["detail"]
        assert not [m for m in items if isinstance(m, Closed)]
        c.say(QUESTION)
        assert one(c.until("agent_message"), "user_message")["message"]["text"] == EN


LONG = "Yes. " + " ".join(f"word{i}" for i in range(15)) + "."  # chunks: "Yes." (0.1 s), then 1.6 s


def test_7_the_playback_fallback_follows_the_audio_as_it_was_sent(voice, monkeypatch):
    monkeypatch.setattr(session_module, "PLAYBACK_GRACE_S", 0.3)
    voice.fakes.llm.reply = LONG
    voice.fakes.tts.delay = 0.8  # slow synthesis: the client waits between chunks
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until(lambda m: isinstance(m, bytes) and parse_frame(m)[1:3] == (1, 0))
        second_chunk = time.monotonic()  # the client starts playing the 1.6 s second chunk now
        c.until(is_state("listening"), timeout=8)
        waited = time.monotonic() - second_chunk
    # gapless arithmetic (first audio + 1.7 s + grace) would end the turn ~0.8 s too early, mid-chunk
    assert waited >= 1.6 + 0.3 - 0.1, waited


def test_7_playback_reports_push_the_fallback_back(voice, monkeypatch):
    monkeypatch.setattr(session_module, "PLAYBACK_GRACE_S", 0.3)
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until("agent_message")  # 1.5 s of audio sent
        sent = time.monotonic()
        time.sleep(1.0)
        c.send("playback", turn_id=1, played_ms=300)  # the client is behind: 1.2 s still to play
        c.until(is_state("listening"), timeout=8)
        waited = time.monotonic() - sent
    assert waited >= 1.0 + 1.2 + 0.3 - 0.1, waited


def test_8_stale_speculative_transcriptions_are_cancelled(voice):
    voice.fakes.stt.delay = 0.5
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        # pause (speculative STT starts), speech resumes (it's stale), pause again, end
        c.audio(speech_audio(500, QUESTION) + silence(400) + speech_audio(500, QUESTION) + silence(700))
        c.until("user_message", timeout=5)
    assert voice.fakes.stt.cancelled == 1  # the stale job didn't run to the end on the single STT thread


def test_9_control_messages_are_handled_between_the_frames_of_a_long_chunk(voice, monkeypatch):
    original = voice_api.StarletteTransport.send_bytes

    async def slow_frames(self, data):
        await asyncio.sleep(0.05)  # a slow link: 50 ms per 200 ms frame
        await original(self, data)

    monkeypatch.setattr(voice_api.StarletteTransport, "send_bytes", slow_frames)
    voice.fakes.llm.reply = "Sure. " + " ".join(f"word{i}" for i in range(29)) + "."  # second chunk: 3 s, 15 frames
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until(lambda m: isinstance(m, dict) and m["type"] == "audio_chunk" and m["chunk_index"] == 1)
        c.send("stop")
        items = c.until(is_state("interrupted"), timeout=5)
    frames_before = [m for m in items if isinstance(m, bytes) and parse_frame(m)[1] == 1]
    assert len(frames_before) < 15  # the stop was handled mid-chunk, not after all its frames


def test_10_pages_from_other_origins_are_refused_with_4403(voice):
    url = f"/ws/chats/{voice.chat}/voice"
    with voice.api.websocket_connect(url, headers={"Origin": "https://evil.example"}) as ws:
        assert VoiceClient(ws).next() == Closed(4403)
    with voice.api.websocket_connect(url, headers={"Origin": "http://localhost:3000"}) as ws:
        assert VoiceClient(ws).start()["type"] == "ready"  # the configured frontend origin
    with voice.connect() as ws:  # no Origin: not a browser page (scripts, tests)
        assert VoiceClient(ws).start()["type"] == "ready"


def test_11_the_espeak_data_link_lives_in_a_private_directory(tmp_path, monkeypatch):
    package = tmp_path / ("deep" * 40) / "espeakng_loader"
    (package / "espeak-ng-data").mkdir(parents=True)
    spec = SimpleNamespace(origin=str(package / "__init__.py"))
    monkeypatch.setattr(speech.importlib.util, "find_spec", lambda name: spec if name == "espeakng_loader" else None)
    monkeypatch.delenv("ESPEAK_DATA_PATH", raising=False)
    speech.shorten_espeak_data_path()
    link = Path(os.environ["ESPEAK_DATA_PATH"])
    assert link.is_symlink() and link.resolve() == (package / "espeak-ng-data").resolve()
    assert link.parent.name.startswith("espeak-") and stat.S_IMODE(link.parent.stat().st_mode) == 0o700
    assert len(str(link)) < speech.ESPEAK_PATH_MAX


# ------------------------------------------------------------------ minor


def test_15_stop_while_a_barge_in_is_being_decided_sends_the_decision(voice):
    voice.fakes.llm.hold_after = HOLD_AFTER_TWO_SENTENCES
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until(chunk_index(1))
        c.send("barge_in_start", turn_id=1, played_ms=700)  # no speech yet: the decision is pending
        c.send("stop")
        items = c.until(is_state("listening"))
    assert kinds(items) == ["barge_in", "state", "agent_message", "state"]
    assert items[0] == {"type": "barge_in", "turn_id": 1, "decision": "stop"}
    assert one(items, "agent_message")["message"]["route"]["interrupted"] == "stop"


def test_16a_stop_after_a_resumed_barge_in_uses_the_current_playback_position(voice):
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until("agent_message")
        c.send("barge_in_start", turn_id=1, played_ms=300)
        c.say(MMHMM, ms=320)
        assert c.until("barge_in")[-1]["decision"] == "resume"
        c.send("playback", turn_id=1, played_ms=1200)
        c.send("stop")
        stopped = one(c.until("agent_message"), "agent_message")["message"]
    # 1200 ms of audio, not the 300 ms of the earlier (resumed) barge-in
    assert stopped["heard_text"] == "The EBITDA margin was 18.2%. Revenue grew 34% in FY24. The board"


def test_16b_a_replaced_session_saves_its_answer_as_disconnected_before_the_new_one_talks(voice):
    voice.fakes.llm.hold_after = HOLD_AFTER_TWO_SENTENCES
    with voice.connect() as ws1:
        first = VoiceClient(ws1)
        first.start()
        first.say(QUESTION)
        first.until(chunk_index(1))
        with voice.connect() as ws2:
            second = VoiceClient(ws2)
            assert first.until(lambda m: isinstance(m, Closed))[-1] == Closed(4409)
            voice.fakes.llm.hold_after = None
            second.start()
            second.say(CORRECTION)
            second.until("agent_message")
    assert [(r[0], r[3]) for r in roles(voice)] == [
        ("user", None),
        ("agent", "disconnect"),  # saved by the first session before the second one's message
        ("user", None),
        ("agent", "disconnect"),  # the second session closed mid-playback too
    ]


def test_17_after_an_llm_error_the_cut_off_fragment_is_not_spoken(voice):
    voice.fakes.llm.fail_with, voice.fakes.llm.fail_after = LLMError("connection reset"), 8  # "… Revenue grew 3"
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        items = c.until(is_state("listening"))
    assert one(items, "error")["stage"] == "llm"
    assert [t for t, _ in voice.fakes.tts.calls] == ["The EBITDA margin was 18.2%."]  # not "Revenue grew 3"


def test_18_websocket_messages_are_limited_to_64_kib(monkeypatch):
    from app import __main__ as main_module

    captured = {}
    monkeypatch.setattr(main_module.uvicorn, "run", lambda *a, **kw: captured.update(kw))
    assert main_module.main() == 0
    assert captured["ws_max_size"] == MAX_MESSAGE_BYTES == 64 * 1024


def test_19_the_next_turn_waits_for_a_cut_turn_whose_decision_is_slow_to_send(voice, monkeypatch):
    original = voice_api.StarletteTransport.send_text

    async def slow_decision(self, text):
        if '"type":"barge_in"' in text:
            await asyncio.sleep(1.0)  # a slow socket write of the decision
        await original(self, text)

    monkeypatch.setattr(voice_api.StarletteTransport, "send_text", slow_decision)
    voice.fakes.llm.hold_after = HOLD_AFTER_TWO_SENTENCES
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until(chunk_index(1))
        voice.fakes.llm.hold_after = None
        c.send("barge_in_start", turn_id=1, played_ms=700)
        c.say(CORRECTION, ms=600)  # ends (and is transcribed) while the decision is still being sent
        items = c.until(lambda m: isinstance(m, dict) and m["type"] == "user_message", timeout=8)
        c.send("stop")
    key = ("barge_in", "agent_message", "user_message")
    order = [m["type"] for m in items if isinstance(m, dict) and m["type"] in key]
    assert order == ["barge_in", "agent_message", "user_message"]
    assert [r[0] for r in roles(voice)][:3] == ["user", "agent", "user"]


# ------------------------------------------------------------------ end-to-end findings


@pytest.mark.parametrize("hum", ["M M", "MM", "Mm-hmm.", "hmm", "उम्म", "हम्म"])
def test_A_a_short_hum_never_cuts_the_answer(voice, hum):
    voice.fakes.stt.scripts[HUM] = lambda ms: hum if ms < 700 else "MM"  # the 250 ms snapshot, then the rest
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until("agent_message")
        c.send("barge_in_start", turn_id=1, played_ms=300)
        c.say(HUM, ms=500)
        items = c.until("barge_in")
        assert items[-1] == {"type": "barge_in", "turn_id": 1, "decision": "resume"}
        rest = c.quiet(0.5)
        assert not of(rest, "user_message") and not of(rest, "agent_message")
        assert rest[-1] == {"type": "state", "state": "speaking"}  # the client is told the answer goes on


def test_B_a_backchannel_without_barge_in_start_restores_the_state(voice):
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until("agent_message")  # speaking; the client's VAD misses the user's "mm-hmm"
        c.say(MMHMM, ms=320)
        items = c.until(is_state("speaking"))
        assert kinds(items) == ["user_speech", "user_speech", "state"]
        assert not of(c.quiet(0.3), "barge_in")  # no barge-in was started, none to resolve


def test_B_a_too_short_burst_restores_the_state_and_resolves_a_pending_barge_in(voice):
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until("agent_message")
        c.send("barge_in_start", turn_id=1, played_ms=300)
        c.say(QUESTION, ms=150)  # under min_speech_ms
        items = c.until(is_state("speaking"))
    assert kinds(items) == ["user_speech", "user_speech", "barge_in", "state"]
    assert one(items, "barge_in")["decision"] == "resume"


# ------------------------------------------------------------------ second end-to-end run

SAID = 75  # what the user says over the answer, set per test as a function of how much audio was transcribed


def paced(c: VoiceClient, pcm: bytes, frame_ms: int = 32) -> None:
    """Send audio in real time, so the server's deadlines fall while the user is still speaking."""
    size = 16 * frame_ms * 2
    start = time.monotonic()
    for n, i in enumerate(range(0, len(pcm), size)):
        time.sleep(max(0.0, start + n * frame_ms / 1000 - time.monotonic()))
        c.ws.send_bytes(pcm[i : i + size])


def speaking_over_the_answer(voice, transcript_of) -> list:
    voice.fakes.stt.scripts[SAID] = transcript_of
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until("agent_message")
        c.send("barge_in_start", turn_id=1, played_ms=300)
        paced(c, speech_audio(1100, SAID) + silence(800))  # still speaking at the 700 ms deadline
        items = c.until("barge_in", timeout=5)
        items += c.quiet(0.6)
    return items


def test_C_yeah_right_still_being_said_at_the_deadline_doesnt_cut_the_answer(voice):
    items = speaking_over_the_answer(voice, lambda ms: "Yeah." if ms < 1000 else "Yeah, right.")
    assert one(items, "barge_in") == {"type": "barge_in", "turn_id": 1, "decision": "resume"}
    assert not of(items, "user_message") and not of(items, "agent_message")  # not answered as a question either
    assert [r[0] for r in roles(voice)] == ["user", "agent"]


def test_C_real_words_after_an_acknowledgement_still_cut_the_answer(voice):
    items = speaking_over_the_answer(voice, lambda ms: "Yeah." if ms < 600 else "Yeah, but what about FY23?")
    assert one(items, "barge_in") == {"type": "barge_in", "turn_id": 1, "decision": "stop"}


def test_C_an_acknowledgement_alone_is_never_answered_as_a_question(voice):
    """While the agent is silent, "Yeah, right." reaches the router's keyword fast path: a short spoken
    acknowledgement, no model call, never "I couldn't find that in the documents"."""
    voice.fakes.stt.scripts[SAID] = "Yeah, right."
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(SAID)
        items = c.until("agent_message")
        agent = one(items, "agent_message")["message"]
        assert agent["route"]["intent"] == "backchannel" and agent["route"]["abstained"] is False
        assert [ch["text"] for ch in of(items, "audio_chunk")] == [agent["text"]]  # a short reply, spoken
    assert voice.fakes.llm.calls == []  # neither the router model nor the answer model was asked
