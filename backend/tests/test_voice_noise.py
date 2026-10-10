"""Noisy rooms (docs/DESIGN.md §3.10): the noise floor, the gate on the endpointer, whether speech was said to the
agent, and hold-to-talk. The session tests use the fakes of test_voice_api.py: the audio's level encodes what was
said (tone t ↔ amplitude t/100, i.e. 20·log10(t/100) dBFS: tone 30 is -10.5 dBFS), FakeVAD hears speech above tone 2.
"""

# ruff: noqa: F811  (the `voice` fixture is imported from test_voice_api and requested by name)

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.api.public_config import public_config
from app.services.voice.noise import (
    DB_MIN,
    QUIET_FLOOR_DBFS,
    NoiseFloor,
    NoiseGate,
    SpeechEvidence,
    UserLevel,
    addressed,
    content_words,
    frame_dbfs,
    is_fragment,
    near_field_min,
    utterance_level,
)
from app.services.voice.protocol import Ptt, SetInputMode, Start, parse_client_message
from app.services.voice.turn_taking import (
    Endpointer,
    SpeechDiscarded,
    SpeechEnded,
    SpeechStarted,
)
from app.settings import ConfigError

from .conftest import LOCAL_CONFIG
from .fakes import silence
from .fakes import speech as speech_audio
from .test_voice_api import (  # noqa: F401  (voice is a fixture)
    EN,
    QUESTION,
    REPLY,
    VoiceClient,
    kinds,
    of,
    one,
    voice,
)

CFG = SimpleNamespace(
    denoise="rnnoise",
    adaptive_gating=True,
    floor_window_ms=8000,
    floor_percentile=20,
    start_snr_db=9,
    end_snr_db=4,
    quiet_floor_dbfs=-60,
    loud_floor_dbfs=-35,
    noisy_threshold=0.8,
    noisy_min_speech_ms=400,
    drop_background_speech=True,
    assumed_user_dbfs=-28,
    assumed_margin_db=4,
    far_field_db=6,
    far_field_hard_db=10,
    close_db=6,
    min_snr_db=6,
    unsure_avg_logprob=-0.65,
    unsure_no_speech_prob=0.5,
    short_fragment_words=3,
)


def level(dbfs: float) -> np.ndarray:
    """A 32 ms frame at ``dbfs`` (a constant: its mean square is exactly the level)."""
    return np.full(512, 10 ** (dbfs / 20), dtype=np.float32)


def frames(ms: float) -> int:
    return round(ms / 32)


def gated(**over: object) -> Endpointer:
    cfg = SimpleNamespace(**{**vars(CFG), **over})
    gate = NoiseGate(cfg, threshold=0.5, min_speech_ms=250)
    return Endpointer(threshold=0.5, min_speech_ms=250, end_of_turn_ms=600, gate=gate)


def feed(ep: Endpointer, *parts: tuple[float, float, float]) -> list[tuple[int, object]]:
    """Feed (ms, dBFS, VAD probability) runs of frames; (frame number, event) for every event."""
    events, i = [], 0
    for ms, dbfs, p in parts:
        for _ in range(frames(ms)):
            events += [(i, e) for e in ep.push(level(dbfs), p)]
            i += 1
    return events


# ------------------------------------------------------------------ levels and the floor


def test_frame_levels():
    assert frame_dbfs(np.zeros(512, dtype=np.float32)) == DB_MIN
    assert frame_dbfs(level(-20)) == pytest.approx(-20, abs=0.01)
    assert frame_dbfs(np.zeros(0, dtype=np.float32)) == DB_MIN


def test_the_floor_is_a_low_percentile_of_the_last_seconds():
    floor = NoiseFloor(window_ms=8000, percentile=20)
    for _ in range(20):
        floor.push(-40)
    assert floor.dbfs == QUIET_FLOOR_DBFS  # under a second: assumed quiet, not measured
    for i in range(200):
        floor.push(-20 if i % 3 == 0 else -50)  # babble: a third of the frames are words
    assert floor.dbfs == pytest.approx(-50)
    for _ in range(300):  # the window moves on: the room got louder
        floor.push(-35)
    assert floor.dbfs == pytest.approx(-35)


def test_the_gate_raises_its_thresholds_with_the_floor():
    gate = NoiseGate(CFG, threshold=0.5, min_speech_ms=250)
    assert (gate.noisiness, gate.threshold, gate.min_speech_ms) == (0, 0.5, 250)  # quiet (assumed)
    for _ in range(100):
        gate.observe(-47.5)  # halfway between quiet (-60) and loud (-35)
    assert gate.noisiness == pytest.approx(0.5)
    assert gate.threshold == pytest.approx(0.65) and gate.min_speech_ms == pytest.approx(325)
    assert not gate.opens(0.9, -40)  # speech by the VAD, but only 7.5 dB above the floor
    assert not gate.opens(0.6, -30)  # loud enough, but under the raised threshold
    assert gate.opens(0.7, -30)
    for _ in range(300):
        gate.observe(-20)
    assert (gate.noisiness, gate.threshold, gate.min_speech_ms) == (1, pytest.approx(0.8), 400)


def test_near_field_is_relative_to_the_user_once_known():
    assert near_field_min(CFG, None) == -42  # assumed -28, 10 dB + 4 dB margin
    assert near_field_min(CFG, -20) == -30
    user = UserLevel()
    gate = NoiseGate(CFG, threshold=0.5, min_speech_ms=250, user=user)
    assert gate.near_field(-40) and gate.near_field(None)
    user.update(-20)
    assert not gate.near_field(-40) and gate.near_field(-29)


def test_the_user_level_rises_fast_and_falls_slowly():
    user = UserLevel()
    user.update(None)
    assert user.dbfs is None
    user.update(-30)
    user.update(-20)
    assert user.dbfs == pytest.approx(-25)
    user.update(-45)  # one quiet turn (or background speech that got through) barely moves it
    assert user.dbfs == pytest.approx(-28)
    assert utterance_level([-50, -40, -30, -20, -10]) == pytest.approx(-18)  # the loud part (80th percentile)
    assert utterance_level([]) is None


# ------------------------------------------------------------------ the endpointer with the gate


def test_noise_the_vad_mistakes_for_speech_opens_no_turn_once_the_floor_is_known():
    ep = gated()
    # 1.5 s of a fan (the VAD isn't fooled), then the same fan with the VAD fooled (babble-like): never above the floor
    assert feed(ep, (1500, -40, 0.1), (3000, -40, 0.9)) == []


def test_near_speech_is_announced_after_min_speech_and_babble_doesnt_hold_it_open():
    ep = gated()
    events = feed(ep, (1500, -60, 0.1), (800, -18, 0.95), (1500, -60, 0.8))
    starts = [i for i, e in events if isinstance(e, SpeechStarted)]
    assert starts == [frames(1500) + frames(250) - 1]  # announced once 250 ms of speech are in, not at its first frame
    ended = [e for _, e in events if isinstance(e, SpeechEnded)]
    assert len(ended) == 1  # ended by the floor-level frames although the VAD still hears "speech" in them
    u = ended[0].utterance
    assert u.level_dbfs == pytest.approx(-18) and u.floor_dbfs == pytest.approx(-60)
    assert u.speech_ms == pytest.approx(frames(800) * 32)


def test_far_speech_is_never_announced():
    ep = gated()
    events = feed(ep, (1500, -80, 0.0), (1000, -50, 0.95), (800, -80, 0.0))
    # the TV across the room (below -42 dBFS, and too quiet to hold a turn open): only quiet discards
    assert events and all(isinstance(e, SpeechDiscarded) and not e.announced for _, e in events)
    assert not ep.in_utterance


def test_far_speech_after_the_user_stops_doesnt_hold_their_turn_open():
    ep = gated()
    ep.gate.user.update(-18)  # type: ignore[union-attr]
    # the user's question, then the TV (20 dB down, the VAD hears speech in it) goes on
    events = feed(ep, (1500, -80, 0.0), (1000, -18, 0.95), (1500, -38, 0.95))
    ended = [i for i, e in events if isinstance(e, SpeechEnded)]
    assert ended == [frames(1500) + frames(1000) + frames(600) - 1]  # 600 ms after the user stopped


def test_a_noisy_room_needs_longer_speech():
    ep = gated()
    events = feed(ep, (2000, -33, 0.1), (300, -15, 0.95), (800, -33, 0.0))  # floor above "loud": 400 ms needed
    assert [type(e) for _, e in events] == [SpeechDiscarded]
    ep = gated(adaptive_gating=False)  # without the gate: announced at once, discarded as too short
    events = feed(ep, (2000, -33, 0.1), (200, -15, 0.95), (800, -33, 0.0))
    assert [type(e) for _, e in events] == [SpeechStarted, SpeechDiscarded]
    assert events[-1][1].announced


# ------------------------------------------------------------------ said to the agent?


def evidence(text: str = "What was the revenue last year?", **over: object) -> SpeechEvidence:
    fields: dict[str, object] = {"level_dbfs": -22.0, "floor_dbfs": -55.0, "user_dbfs": -22.0, **over}
    return SpeechEvidence(text, **fields)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("case", "verdict"),
    [
        (evidence(), "answer"),  # the user, clearly
        (evidence(garbled=True, avg_logprob=-1.3), "say_again"),  # the user, garbled: ask again
        (evidence(level_dbfs=-33.0), "drop"),  # 11 dB below the user: the TV, however clear
        (evidence(level_dbfs=-30.0), "answer"),  # 8 dB below, clear and with content: still answered
        (evidence(level_dbfs=-30.0, avg_logprob=-0.9), "drop"),  # 8 dB below and unsure
        (evidence("and then he", level_dbfs=-30.0), "drop"),  # 8 dB below, a fragment
        (evidence(level_dbfs=-30.0, floor_dbfs=-33.0), "drop"),  # 8 dB below and weak against the floor
        (evidence(level_dbfs=-30.0, garbled=True), "drop"),  # garbled far speech: not "say again"
        (evidence("Thank you.", floor_dbfs=-25.0, user_dbfs=None, level_dbfs=-30.0), "drop"),  # weak, a fragment
        (evidence(floor_dbfs=-25.0, user_dbfs=-24.0, avg_logprob=-0.9), "answer"),  # loud café, the user's level
        (evidence(floor_dbfs=-25.0, user_dbfs=-24.0, garbled=True), "say_again"),
        (evidence(user_dbfs=None, level_dbfs=-38.0), "answer"),  # first turn, quiet but clear: answered
        (evidence(user_dbfs=None, level_dbfs=-43.0), "drop"),  # first turn, below the assumed near field
        (evidence("Stop.", level_dbfs=-30.0), "answer"),  # a stop cue is never a fragment
        (evidence("Thank you.", user_dbfs=None, avg_logprob=-0.9), "drop"),  # Whisper's word for café clatter
        (evidence("Thank you.", avg_logprob=-0.9), "answer"),  # ... but said at the user's own level
        (evidence("Thank you.", user_dbfs=None), "answer"),  # a clear "thank you" before any turn
        (evidence(level_dbfs=None), "answer"),  # no level: decided on the words alone
        (evidence("   "), "drop"),
    ],
)
def test_addressed(case, verdict):
    assert addressed(case, CFG).verdict == verdict


def test_addressed_without_the_background_check_is_the_old_say_again_rule():
    off = SimpleNamespace(**{**vars(CFG), "drop_background_speech": False})
    assert addressed(evidence(level_dbfs=-60.0), off).verdict == "answer"
    assert addressed(evidence(level_dbfs=-60.0, garbled=True), off).verdict == "say_again"


@pytest.mark.parametrize(
    ("text", "content", "fragment"),
    [
        ("What was the revenue?", 1, False),
        ("and then he", 0, True),
        ("Thank you.", 0, True),
        ("Okay.", 0, True),
        ("है ना", 0, True),
        ("मुनाफा कितना था?", 2, False),
        ("FY24", 1, False),
        ("stop", 1, False),
        ("बस", 1, False),  # a stop cue
        ("so I think that we should go", 0, False),  # no content, but too long for a fragment
    ],
)
def test_content_words_and_fragments(text, content, fragment):
    assert content_words(text) == content
    assert is_fragment(text, 3) is fragment


# ------------------------------------------------------------------ protocol and config


def test_hold_to_talk_messages():
    assert parse_client_message('{"type": "ptt", "state": "down"}') == Ptt(type="ptt", state="down")
    assert parse_client_message('{"type": "input_mode", "mode": "ptt"}') == SetInputMode(type="input_mode", mode="ptt")
    assert parse_client_message('{"type": "start", "language": null}') == Start(type="start", input_mode="vad")
    assert parse_client_message('{"type": "start", "input_mode": "ptt"}').input_mode == "ptt"  # type: ignore[union-attr]


def test_noise_settings(load_local):
    n = load_local().voice.noise
    assert (n.denoise, n.adaptive_gating, n.drop_background_speech) == ("rnnoise", True, True)
    assert (n.far_field_db, n.far_field_hard_db, n.assumed_user_dbfs, n.assumed_margin_db) == (6, 8, -26, 2)
    assert load_local(VOICE__NOISE__DENOISE="off").voice.noise.denoise == "off"
    with pytest.raises(ConfigError, match=r"voice\.noise\.denoise"):
        load_local(VOICE__NOISE__DENOISE="speex")
    with pytest.raises(ConfigError, match="loud_floor_dbfs"):
        load_local(VOICE__NOISE__LOUD_FLOOR_DBFS="-70")
    with pytest.raises(ConfigError, match="far_field_hard_db"):
        load_local(VOICE__NOISE__FAR_FIELD_HARD_DB="3")
    with pytest.raises(ConfigError, match=r"voice\.noise\.floor_percentile"):
        load_local(VOICE__NOISE__FLOOR_PERCENTILE="100")
    raw = json.loads(LOCAL_CONFIG.read_text())
    assert set(raw["voice"]["noise"]) == set(type(n).model_fields)


def test_the_browser_gets_the_same_thresholds(load_local):
    s = load_local()
    v = public_config(s).voice_input
    n = s.voice.noise
    assert (v.denoise, v.threshold, v.min_speech_ms) == ("rnnoise", s.vad.threshold, s.vad.min_speech_ms)
    assert (v.start_snr_db, v.noisy_threshold, v.far_field_hard_db) == (n.start_snr_db, n.noisy_threshold, 8)


# ------------------------------------------------------------------ the session

FAR, UNSURE_FAR = 7, 15  # -23 dBFS: 12.5 dB below a user at tone 30 (-10.5); -16.5 dBFS: 6 dB below


def asked(c: VoiceClient) -> None:
    """The user asks one question (tone 30) and hears all of the answer: their level is known from here on."""
    c.say(QUESTION)
    c.until("agent_message")
    c.send("playback_done", turn_id=1)
    c.until(lambda m: m == {"type": "state", "state": "listening"})


def test_far_speech_after_the_users_turn_never_reaches_the_client(voice):
    voice.fakes.stt.scripts[FAR] = "And in other news, the city council approved the budget."
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        asked(c)
        calls = len(voice.fakes.stt.calls)
        c.say(FAR, ms=1500)
        assert c.quiet(0.6) == []  # no user_speech, no turn, no "say again"
    assert len(voice.fakes.stt.calls) == calls  # not even transcribed
    assert [m["role"] for m in voice.transcript()] == ["user", "agent"]


def test_background_speech_is_dropped_silently_not_asked_again(voice):
    voice.fakes.stt.scripts[UNSURE_FAR] = "We really love you over there honestly."
    voice.fakes.stt.confidence[UNSURE_FAR] = {"avg_logprob": -1.3}  # garbled by Whisper's own measure
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        asked(c)
        c.say(UNSURE_FAR, ms=800)
        items = c.until(lambda m: m == {"type": "state", "state": "listening"})
        assert [m["phase"] for m in of(items, "user_speech")] == ["start", "end"]  # it was heard, then dropped
        assert not of(items + c.quiet(0.4), "user_message")
    assert [m["role"] for m in voice.transcript()] == ["user", "agent"]  # neither answered nor "say again"


def test_background_speech_over_the_answer_doesnt_stop_it(voice):
    voice.fakes.stt.scripts[UNSURE_FAR] = "No wait, the weather forecast is after the break."
    voice.fakes.stt.confidence[UNSURE_FAR] = {"avg_logprob": -0.9}
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        asked(c)
        c.say(QUESTION)
        c.until("agent_message")
        c.send("barge_in_start", turn_id=2, played_ms=300)
        c.say(UNSURE_FAR, ms=800)
        items = c.until("barge_in")
        assert items[-1] == {"type": "barge_in", "turn_id": 2, "decision": "resume"}
        assert not of(c.quiet(0.4), "user_message")
        c.send("playback_done", turn_id=2)
        c.until(lambda m: m == {"type": "state", "state": "listening"})
    assert voice.transcript()[-1]["heard_text"] is None  # the answer went on, heard in full


def test_a_near_field_garbled_question_is_still_asked_again(voice):
    voice.fakes.stt.confidence[QUESTION] = {"avg_logprob": -1.4}
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        agent = one(c.until("agent_message"), "agent_message")["message"]
    assert agent["text"].startswith("Sorry, I didn't catch that.")


# ------------------------------------------------------------------ hold-to-talk


def test_hold_to_talk_turns_are_the_audio_between_down_and_up(voice):
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.send("start", language="en", input_mode="ptt")
        assert one([c.next()], "ready")
        c.until("state")
        c.say(QUESTION)  # speaking without the button: nothing
        assert c.quiet(0.5) == []
        c.send("ptt", state="down")
        assert c.next() == {"type": "user_speech", "phase": "start"}
        c.audio(speech_audio(600, QUESTION) + silence(200))
        c.send("ptt", state="up")
        items = c.until("agent_message")
        assert kinds(items)[:4] == ["user_speech", "state", "user_message", "turn"]
        assert one(items, "user_message")["message"]["text"] == EN
        assert one(items, "agent_message")["message"]["text"] == REPLY


def test_pressing_to_talk_stops_the_answer_at_once(voice):
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.send("input_mode", mode="ptt")
        c.send("ptt", state="down")
        c.audio(speech_audio(600, QUESTION))
        c.send("ptt", state="up")
        c.until("agent_message")
        c.send("barge_in_start", turn_id=1, played_ms=200)  # the VAD doesn't interrupt in this mode
        assert c.until("barge_in")[-1] == {"type": "barge_in", "turn_id": 1, "decision": "resume"}
        c.send("ptt", state="down")
        items = c.until("agent_message")
        assert one(items, "barge_in") == {"type": "barge_in", "turn_id": 1, "decision": "stop"}
        assert items[0] == {"type": "user_speech", "phase": "start"}
        assert one(items, "agent_message")["message"]["route"]["interrupted"] == "barge_in"
        c.send("ptt", state="up")  # nothing said: ignored
        assert c.until(lambda m: isinstance(m, dict) and m["type"] == "user_speech")[-1]["phase"] == "end"
        assert not of(c.quiet(0.4), "user_message")


def test_switching_back_to_vad_drops_a_held_press(voice):
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.audio(silence(1500))  # the room before anyone talks (the fake's speech is one unbroken level)
        c.send("ptt", state="down")  # "down" switches to hold-to-talk by itself
        assert c.next() == {"type": "user_speech", "phase": "start"}
        c.audio(speech_audio(600, QUESTION))
        c.send("input_mode", mode="vad")
        items = c.until("state")
        assert of(items, "user_speech") == [{"type": "user_speech", "phase": "end"}]
        assert not of(c.quiet(0.4), "user_message")
        c.say(QUESTION)  # the VAD opens turns again
        assert one(c.until("user_message"), "user_message")["message"]["text"] == EN
