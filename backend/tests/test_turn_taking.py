"""Endpointing, the barge-in verdict and the binary frame format (app/services/voice/turn_taking.py, protocol.py)."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.voice.protocol import (
    HEADER,
    ProtocolError,
    Start,
    audio_frames,
    parse_client_message,
    parse_frame,
)
from app.services.voice.turn_taking import (
    BargeInEvidence,
    Endpointer,
    SpeechDiscarded,
    SpeechEnded,
    SpeechPaused,
    SpeechResumed,
    SpeechStarted,
    barge_in_verdict,
)

FRAME_MS = 32


def run(ep: Endpointer, probabilities: list[float]) -> list[tuple[int, object]]:
    """Feed one 32 ms frame per probability; (frame number, event) for every event."""
    events = []
    for i, p in enumerate(probabilities):
        frame = np.full(512, i, dtype=np.float32)  # frame i holds the value i, to check which audio is kept
        events += [(i, e) for e in ep.push(frame, p)]
    return events


def frames(ms: float) -> int:
    return round(ms / FRAME_MS)


@pytest.fixture
def ep() -> Endpointer:
    return Endpointer(threshold=0.5, min_speech_ms=250, end_of_turn_ms=600)


def test_an_utterance_ends_after_the_end_of_turn_silence(ep):
    probs = [0.0] * 10 + [0.9] * frames(1000) + [0.0] * frames(700)
    events = run(ep, probs)
    kinds = [type(e).__name__ for _, e in events]
    assert kinds == ["SpeechStarted", "SpeechPaused", "SpeechEnded"]
    (started, _), (paused_at, paused), (ended_at, ended) = events
    assert started == 10
    assert (paused_at - 10 - frames(1000) + 1) * FRAME_MS >= 300  # speculative STT at half the end-of-turn wait
    assert isinstance(paused, SpeechPaused) and isinstance(ended, SpeechEnded)
    assert 600 <= (ended_at - 10 - frames(1000) + 1) * FRAME_MS < 640
    utt = ended.utterance
    assert utt.speech_ms == pytest.approx(frames(1000) * FRAME_MS)
    assert utt.silence_ms >= 600
    kept = utt.audio.reshape(-1, 512)[:, 0]
    assert kept[0] == 4 and 10 in kept  # ~192 ms of pre-roll before the detected start
    assert len(paused.audio) < len(utt.audio)


def test_hysteresis_keeps_speech_on_between_the_thresholds(ep):
    # 0.4 is below the 0.5 "on" threshold but above the 0.35 "off" one: no silence starts
    events = run(ep, [0.9] * 10 + [0.4] * frames(2000) + [0.0] * frames(700))
    ended = [e for _, e in events if isinstance(e, SpeechEnded)]
    assert len(ended) == 1 and ended[0].utterance.speech_ms > 2000


def test_a_short_pause_does_not_split_the_turn_and_resuming_invalidates_the_speculative_transcript(ep):
    probs = [0.9] * frames(500) + [0.0] * frames(400) + [0.9] * frames(500) + [0.0] * frames(700)
    kinds = [type(e).__name__ for _, e in run(ep, probs)]
    assert kinds == ["SpeechStarted", "SpeechPaused", "SpeechResumed", "SpeechPaused", "SpeechEnded"]


def test_short_bursts_are_discarded(ep):
    events = run(ep, [0.9] * frames(200) + [0.0] * frames(700))
    assert [type(e) for _, e in events] == [SpeechStarted, SpeechDiscarded]
    assert events[-1][1].speech_ms < 250


def test_state_for_barge_in_decisions(ep):
    run(ep, [0.9] * frames(320))
    assert ep.in_utterance and ep.speaking and ep.speech_ms == pytest.approx(320, abs=FRAME_MS)
    run(ep, [0.0] * 3)
    assert ep.in_utterance and not ep.speaking
    assert len(ep.snapshot()) == (frames(320) + 3) * 512


def test_very_long_speech_is_cut_into_turns():
    ep = Endpointer(threshold=0.5, min_speech_ms=250, end_of_turn_ms=600, max_utterance_ms=1000)
    events = run(ep, [0.9] * frames(2100))
    assert sum(isinstance(e, SpeechEnded) for _, e in events) == 2


def test_resumed_is_only_reported_after_a_pause(ep):
    events = run(ep, [0.9] * frames(500) + [0.0] * 3 + [0.9] * 5)
    assert not any(isinstance(e, SpeechResumed) for _, e in events)


# ------------------------------------------------------------------ barge-in verdict


def evidence(**kw) -> BargeInEvidence:
    fields = {"speech_ms": 0.0, "speaking": False, "transcript": None, "ended": False, "deadline_passed": False}
    return BargeInEvidence(**{**fields, **kw})


@pytest.mark.parametrize(
    ("kw", "backchannel", "verdict"),
    [
        ({"speech_ms": 300, "speaking": True, "transcript": "No wait"}, False, "stop"),  # real words: stop at once
        ({"speech_ms": 300, "speaking": True, "transcript": "Mm-hmm"}, True, None),  # wait for more
        ({"speech_ms": 300, "transcript": "Mm-hmm", "ended": True}, True, "resume"),
        ({"speech_ms": 100, "deadline_passed": True}, None, "resume"),  # too short: noise
        ({"speech_ms": 600, "speaking": True, "deadline_passed": True}, None, "stop"),  # still talking
        # still talking but only acknowledgements so far ("Yeah…" of "Yeah, right"): wait, then resume at the cap
        ({"speech_ms": 600, "speaking": True, "transcript": "Yeah.", "deadline_passed": True}, True, None),
        (
            {
                "speech_ms": 900,
                "speaking": True,
                "transcript": "Yeah, right",
                "deadline_passed": True,
                "cap_passed": True,
            },
            True,
            "resume",
        ),
        ({"speech_ms": 900, "speaking": True, "transcript": "Yeah, but FY23?", "deadline_passed": True}, False, "stop"),
        ({"speech_ms": 350, "speaking": False, "deadline_passed": True}, None, "resume"),  # short burst, over
        ({"speech_ms": 300, "speaking": True}, None, None),
    ],
)
def test_barge_in_verdict(kw, backchannel, verdict):
    kw.setdefault("real_words", len((kw.get("transcript") or "").split()))
    assert barge_in_verdict(evidence(**kw), min_speech_ms=250, is_backchannel=backchannel) == verdict


@pytest.mark.parametrize(
    ("kw", "verdict"),
    [
        # a one-word partial of speech that has already stopped: maybe a misheard "mm-hmm" — wait, don't stop yet
        ({"speech_ms": 300, "speaking": False, "transcript": "No", "real_words": 1}, None),
        ({"speech_ms": 300, "speaking": False, "transcript": "No", "real_words": 1, "deadline_passed": True}, "resume"),
        # ... but the same word while the user is still talking, or two real words, stop at once
        ({"speech_ms": 300, "speaking": True, "transcript": "No", "real_words": 1}, "stop"),
        ({"speech_ms": 300, "speaking": False, "transcript": "No wait", "real_words": 2}, "stop"),
    ],
)
def test_short_partials_of_finished_speech_wait_for_more_evidence(kw, verdict):
    assert barge_in_verdict(evidence(**kw), min_speech_ms=250, is_backchannel=False) == verdict


@pytest.mark.parametrize(
    ("kw", "verdict"),
    [
        # B3: still talking at the deadline with the snapshot's transcription still running: wait for it ...
        ({"speech_ms": 600, "speaking": True, "deadline_passed": True, "transcribing": True}, None),
        # ... up to the acknowledgement cap, then (nothing heard yet) stop as before
        (
            {"speech_ms": 1200, "speaking": True, "deadline_passed": True, "transcribing": True, "cap_passed": True},
            "stop",
        ),
        ({"speech_ms": 600, "speaking": True, "deadline_passed": True, "transcribing": False}, "stop"),
        # a transcription running doesn't hold back a decision the evidence already makes
        ({"speech_ms": 100, "speaking": True, "deadline_passed": True, "transcribing": True}, "resume"),
        # speech that stopped with a fuller transcript coming ("Yeah… but what about FY23?"): wait for it, at most
        # until the cap
        ({"speech_ms": 600, "speaking": False, "deadline_passed": True, "transcribing": True}, None),
        (
            {"speech_ms": 600, "speaking": False, "deadline_passed": True, "transcribing": True, "cap_passed": True},
            "resume",
        ),
    ],
)
def test_a_transcription_still_running_at_the_deadline_is_waited_for_until_the_cap(kw, verdict):
    assert barge_in_verdict(evidence(**kw), min_speech_ms=250, is_backchannel=None) == verdict


# ------------------------------------------------------------------ protocol


def test_audio_frames_carry_turn_chunk_and_sequence():
    pcm = np.arange(24_000 * 450 // 1000, dtype="<i2").tobytes()  # 450 ms at 24 kHz
    frames_ = audio_frames(7, 3, pcm, sample_rate=24_000)
    assert len(frames_) == 3  # 200 + 200 + 50 ms
    assert frames_[0][:12] == bytes.fromhex("070000000300000000000000")  # three little-endian uint32
    parsed = [parse_frame(f) for f in frames_]
    assert [(t, c, s) for t, c, s, _ in parsed] == [(7, 3, 0), (7, 3, 1), (7, 3, 2)]
    assert b"".join(p for *_, p in parsed) == pcm
    assert HEADER.size == 12


def test_client_messages_are_validated():
    assert parse_client_message('{"type": "start", "language": null}') == Start(type="start", language=None)
    assert parse_client_message('{"type": "start", "language": "hi", "extra": 1}').language == "hi"
    for bad in ("nope", '{"type": "dance"}', '{"type": "start", "language": "fr"}', '{"type": "barge_in_start"}'):
        with pytest.raises(ProtocolError):
            parse_client_message(bad)
