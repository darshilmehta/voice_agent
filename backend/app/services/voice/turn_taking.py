"""Turn-taking on the server's VAD (docs/DESIGN.md §3.3 c-d, §9.4): utterance endpointing and the barge-in verdict.

Endpointing follows Silero's own streaming logic (``VADIterator``), on audio time rather than wall time:

    speech starts   at the first frame with p ≥ threshold (the utterance keeps PRE_ROLL_MS of audio before it)
    silence starts  at the first frame with p < threshold - 0.15 and is reset by a frame with p ≥ threshold
    pause           after end_of_turn_ms / 2 of silence: the audio so far is final unless speech resumes, so
                    speculative STT starts here (§9.4: end-of-turn costs 0.65-1.05 s)
    end of turn     after end_of_turn_ms of silence (or MAX_UTTERANCE_MS of audio); utterances with less than
                    min_speech_ms of speech are discarded

Everything here is synchronous and pure, so it is tested without audio or models.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512
OFF_MARGIN = 0.15  # Silero's hysteresis: speech ends below threshold - 0.15
PRE_ROLL_MS = 192  # audio kept before the detected start (Whisper needs the onset)
MAX_UTTERANCE_MS = 30_000  # Whisper's window; longer speech is cut into several turns


@dataclass(frozen=True, slots=True)
class Utterance:
    audio: np.ndarray  # float32, 16 kHz: pre-roll + speech + trailing silence
    speech_ms: float  # from the detected start to the start of the trailing silence
    silence_ms: float  # trailing silence included at the end


@dataclass(frozen=True, slots=True)
class SpeechStarted:
    pass


@dataclass(frozen=True, slots=True)
class SpeechPaused:
    """Silence reached half the end-of-turn wait: ``audio`` is the utterance unless speech resumes."""

    audio: np.ndarray
    speech_ms: float


@dataclass(frozen=True, slots=True)
class SpeechResumed:
    """Speech after a pause: a speculative transcript of the paused audio is stale."""


@dataclass(frozen=True, slots=True)
class SpeechEnded:
    utterance: Utterance


@dataclass(frozen=True, slots=True)
class SpeechDiscarded:
    """The utterance ended with less than min_speech_ms of speech (a click, a cough): ignored."""

    speech_ms: float


VADEvent = SpeechStarted | SpeechPaused | SpeechResumed | SpeechEnded | SpeechDiscarded


class Endpointer:
    def __init__(
        self,
        *,
        threshold: float,
        min_speech_ms: float,
        end_of_turn_ms: float,
        sample_rate: int = SAMPLE_RATE,
        pre_roll_ms: float = PRE_ROLL_MS,
        max_utterance_ms: float = MAX_UTTERANCE_MS,
    ) -> None:
        self.on = threshold
        self.off = max(threshold - OFF_MARGIN, 0.01)
        self.min_speech_ms = min_speech_ms
        self.end_of_turn_ms = end_of_turn_ms
        self.pause_ms = end_of_turn_ms / 2
        self.sample_rate = sample_rate
        self.max_utterance_ms = max_utterance_ms
        self._pre_roll_ms = pre_roll_ms
        self._pre: deque[np.ndarray] = deque()
        self._pre_samples = 0
        self._frames: list[np.ndarray] | None = None  # None: no utterance open
        self._samples = 0  # samples since the detected start
        self._silence = 0  # samples of trailing silence
        self._paused = False

    # -------------------------------------------------------------- state

    @property
    def in_utterance(self) -> bool:
        return self._frames is not None

    @property
    def speaking(self) -> bool:
        """Inside an utterance and not in its trailing silence."""
        return self._frames is not None and self._silence == 0

    @property
    def speech_ms(self) -> float:
        """Speech so far in the open utterance (0 when none is open)."""
        return self._ms(self._samples - self._silence) if self._frames is not None else 0.0

    def snapshot(self) -> np.ndarray:
        """The open utterance's audio so far (empty when none is open)."""
        return np.concatenate(self._frames) if self._frames else np.zeros(0, dtype=np.float32)

    def _ms(self, samples: int) -> float:
        return samples * 1000 / self.sample_rate

    # -------------------------------------------------------------- frames

    def push(self, frame: np.ndarray, probability: float) -> list[VADEvent]:
        n = len(frame)
        if self._frames is None:
            if probability >= self.on:
                self._frames = [*self._pre, frame]
                self._pre.clear()
                self._pre_samples = 0
                self._samples, self._silence, self._paused = n, 0, False
                return [SpeechStarted()]
            self._remember(frame)
            return []

        self._frames.append(frame)
        self._samples += n
        events: list[VADEvent] = []
        if probability >= self.on:
            if self._silence and self._paused:
                events.append(SpeechResumed())
            self._silence, self._paused = 0, False
        elif probability < self.off or self._silence:
            self._silence += n

        silence_ms = self._ms(self._silence)
        if silence_ms >= self.end_of_turn_ms or self._ms(self._samples) >= self.max_utterance_ms:
            events.append(self._close())
        elif not self._paused and silence_ms >= self.pause_ms and self.speech_ms >= self.min_speech_ms:
            self._paused = True
            events.append(SpeechPaused(self.snapshot(), self.speech_ms))
        return events

    def _remember(self, frame: np.ndarray) -> None:
        self._pre.append(frame)
        self._pre_samples += len(frame)
        while self._pre and self._ms(self._pre_samples - len(self._pre[0])) >= self._pre_roll_ms:
            self._pre_samples -= len(self._pre.popleft())

    def _close(self) -> SpeechEnded | SpeechDiscarded:
        frames = self._frames or []
        speech_ms, silence_ms = self.speech_ms, self._ms(self._silence)
        self._frames, self._samples, self._silence, self._paused = None, 0, 0, False
        for frame in frames[-max(1, round(self._pre_roll_ms * self.sample_rate / 1000 / FRAME_SAMPLES)) :]:
            self._remember(frame)  # the trailing silence is the next utterance's pre-roll
        if speech_ms < self.min_speech_ms:
            return SpeechDiscarded(speech_ms)
        return SpeechEnded(Utterance(np.concatenate(frames), speech_ms, silence_ms))

    def reset(self) -> None:
        self._pre.clear()
        self._pre_samples = 0
        self._frames, self._samples, self._silence, self._paused = None, 0, 0, False


# ------------------------------------------------------------------ barge-in


@dataclass(frozen=True, slots=True)
class BargeInEvidence:
    """What the server knows about speech that started while the agent was talking."""

    speech_ms: float  # server-VAD speech so far in the interrupting utterance
    speaking: bool  # the user is still talking (not in trailing silence)
    transcript: str | None  # latest transcript of that speech (partial or final), None if not available yet
    ended: bool  # the utterance is over (end of turn, final transcript known)
    deadline_passed: bool  # decision_timeout_ms since barge_in_start
    real_words: int = 0  # words of the transcript that are neither hums ("M M") nor Whisper noise


Verdict = Literal["stop", "resume"]


def barge_in_verdict(evidence: BargeInEvidence, *, min_speech_ms: float, is_backchannel: bool | None) -> Verdict | None:
    """Duck, then decide (§3.3 c). ``is_backchannel`` is the transcript's classification (None without a transcript;
    hums, fillers, noise and empty transcripts are backchannels).

    1. a transcript that isn't a backchannel                          → stop (as soon as it is known), except a
       partial of fewer than 2 real words of speech that has already stopped: Whisper often mishears a short
       snapshot ("M M" for "mm-hmm"), so that waits for the deadline or the final transcript
    2. the utterance ended as a backchannel, noise or too short        → resume
    3. before the deadline                                             → wait (None)
    4. at the deadline: speech shorter than min_speech_ms              → resume
                        the user is still talking                      → stop (backchannels are short)
                        a short burst that has stopped                 → resume; if its final transcript turns out not
                                                                         to be a backchannel, the session still stops
                                                                         the answer when the utterance ends
    """
    sure = evidence.speaking or evidence.ended or evidence.real_words >= 2
    if evidence.transcript is not None and is_backchannel is False and sure:
        return "stop"
    if evidence.ended:
        return "resume"
    if not evidence.deadline_passed:
        return None
    if evidence.speech_ms < min_speech_ms:
        return "resume"
    return "stop" if evidence.speaking else "resume"
