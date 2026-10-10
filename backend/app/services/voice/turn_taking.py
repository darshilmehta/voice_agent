"""Turn-taking on the server's VAD (docs/DESIGN.md §3.3 c-d, §9.4): utterance endpointing and the barge-in verdict.

Endpointing follows Silero's own streaming logic (``VADIterator``), on audio time rather than wall time:

    speech starts   at the first frame with p ≥ threshold (the utterance keeps PRE_ROLL_MS of audio before it)
    silence starts  at the first frame with p < threshold - 0.15 and is reset by a frame with p ≥ threshold
    pause           after end_of_turn_ms / 2 of silence: the audio so far is final unless speech resumes, so
                    speculative STT starts here (§9.4: end-of-turn costs 0.65-1.05 s)
    end of turn     after end_of_turn_ms of silence (or MAX_UTTERANCE_MS of audio); utterances with less than
                    min_speech_ms of speech are discarded

With a ``NoiseGate`` (voice.noise.adaptive_gating, §3.10 "Noisy rooms") the room's noise floor takes part: speech
starts only on a frame that is also start_snr_db above the floor, at a VAD threshold and a minimum speech length that
rise with the floor, and a frame within end_snr_db of the floor is silence whatever the VAD says (babble alone can't
hold a turn open); a turn is announced only once it has min_speech_ms of speech near the user's level (the gate's
``near_field``), so far-off talk never reaches the client at all. Every utterance carries its speech level and the
floor when it began, for ``noise.addressed``.

Everything here is synchronous and pure, so it is tested without audio or models.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .noise import NoiseGate, frame_dbfs, utterance_level

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
    level_dbfs: float | None = None  # its speech level: the median level of its speech frames
    floor_dbfs: float | None = None  # the room's noise floor when it began (None without a NoiseGate)


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
    """The utterance ended with less than min_speech_ms of speech (a click, a cough), or (with the noise gate) never
    sounded near enough to be the user's: ignored. ``announced``: its SpeechStarted was sent (only then does the client
    need a ``user_speech end``)."""

    speech_ms: float
    announced: bool = True


VADEvent = SpeechStarted | SpeechPaused | SpeechResumed | SpeechEnded | SpeechDiscarded


class Endpointer:
    """Utterances from VAD probabilities (module docstring). With an adaptive ``NoiseGate`` an utterance opens on a
    frame above the room's floor but is only *announced* (SpeechStarted) once it has min_speech of speech at a level
    near the user's (``NoiseGate.near_field``): until then it is held quietly, and if it never gets there it ends as an
    unannounced SpeechDiscarded (the TV, babble, a cough: the client never hears about it). Without the gate every
    utterance is announced at its first frame, as before."""

    def __init__(
        self,
        *,
        threshold: float,
        min_speech_ms: float,
        end_of_turn_ms: float,
        sample_rate: int = SAMPLE_RATE,
        pre_roll_ms: float = PRE_ROLL_MS,
        max_utterance_ms: float = MAX_UTTERANCE_MS,
        gate: NoiseGate | None = None,
    ) -> None:
        self.gate = gate  # the noise floor (always tracked when given) and, if adaptive, the gate on it
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
        self._confirmed = False  # the open utterance was announced
        self._samples = 0  # samples since the detected start
        self._silence = 0  # samples of trailing silence
        self._paused = False
        self._levels: list[float] = []  # levels of the open utterance's speech frames
        self._floor_at_start: float | None = None
        self.last_level: float | None = None  # the last closed utterance's speech level and floor
        self.last_floor: float | None = None

    # -------------------------------------------------------------- state

    @property
    def in_utterance(self) -> bool:
        """An announced utterance is open (one still held by the gate doesn't count)."""
        return self._frames is not None and self._confirmed

    @property
    def speaking(self) -> bool:
        """Inside an announced utterance and not in its trailing silence."""
        return self.in_utterance and self._silence == 0

    @property
    def speech_ms(self) -> float:
        """Speech so far in the announced utterance (0 when none is open)."""
        return self._speech_ms() if self.in_utterance else 0.0

    def _speech_ms(self) -> float:
        return self._ms(self._samples - self._silence) if self._frames is not None else 0.0

    def snapshot(self) -> np.ndarray:
        """The open utterance's audio so far (empty when none is open)."""
        return np.concatenate(self._frames) if self._frames else np.zeros(0, dtype=np.float32)

    @property
    def level_dbfs(self) -> float | None:
        """The open utterance's speech level so far (None when none is open or it has no speech frames yet)."""
        return utterance_level(self._levels) if self._frames is not None else None

    @property
    def floor_at_start(self) -> float | None:
        """The noise floor when the open utterance began."""
        return self._floor_at_start if self._frames is not None else None

    @property
    def floor_dbfs(self) -> float | None:
        return self.gate.floor_dbfs if self.gate is not None else None

    @property
    def gating(self) -> bool:
        return self.gate is not None and self.gate.cfg.adaptive_gating

    @property
    def min_speech(self) -> float:
        """The speech an utterance needs (min_speech_ms, raised in a noisy room by the gate)."""
        return max(self.min_speech_ms, self.gate.min_speech_ms) if self.gating and self.gate else self.min_speech_ms

    def _ms(self, samples: int) -> float:
        return samples * 1000 / self.sample_rate

    # -------------------------------------------------------------- frames

    def observe(self, frame: np.ndarray) -> None:
        """A frame that isn't listened to for speech (hold-to-talk mode): only the floor and the pre-roll follow it."""
        if self.gate is not None:
            self.gate.observe(frame_dbfs(frame))
        if self._frames is None:
            self._remember(frame)

    def push(self, frame: np.ndarray, probability: float) -> list[VADEvent]:
        n = len(frame)
        gate = self.gate
        dbfs = frame_dbfs(frame) if gate is not None else 0.0
        if gate is not None:
            gate.observe(dbfs)
        gating = self.gating and gate is not None
        if self._frames is None:
            starts = gate.opens(probability, dbfs) if gating and gate is not None else probability >= self.on
            if starts:
                self._frames = [*self._pre, frame]
                self._pre.clear()
                self._pre_samples = 0
                self._samples, self._silence, self._paused = n, 0, False
                self._levels = [dbfs] if gate is not None else []
                self._floor_at_start = gate.floor_dbfs if gate is not None else None
                self._confirmed = not gating
                return [SpeechStarted()] if self._confirmed else self._confirm()
            self._remember(frame)
            return []

        self._frames.append(frame)
        self._samples += n
        events: list[VADEvent] = []
        audible = gate.audible(dbfs) and gate.holds(dbfs) if gating and gate is not None else True
        if probability >= self.on and audible:
            if self._silence and self._paused:
                events.append(SpeechResumed())
            self._silence, self._paused = 0, False
            if gate is not None:
                self._levels.append(dbfs)
        elif probability < self.off or self._silence or not audible:
            self._silence += n
        if not self._confirmed:
            events += self._confirm()

        silence_ms = self._ms(self._silence)
        if silence_ms >= self.end_of_turn_ms or self._ms(self._samples) >= self.max_utterance_ms:
            events.append(self._close())
        elif self._confirmed and not self._paused and silence_ms >= self.pause_ms and self.speech_ms >= self.min_speech:
            self._paused = True
            events.append(SpeechPaused(self.snapshot(), self.speech_ms))
        return events

    def _confirm(self) -> list[VADEvent]:
        """Announce the held utterance once it has min_speech of speech near the user's level."""
        gate = self.gate
        if gate is None or self._speech_ms() < self.min_speech or self._silence:
            return []
        if not gate.near_field(self.level_dbfs):
            return []
        self._confirmed = True
        return [SpeechStarted()]

    def _remember(self, frame: np.ndarray) -> None:
        self._pre.append(frame)
        self._pre_samples += len(frame)
        while self._pre and self._ms(self._pre_samples - len(self._pre[0])) >= self._pre_roll_ms:
            self._pre_samples -= len(self._pre.popleft())

    def _close(self) -> SpeechEnded | SpeechDiscarded:
        frames = self._frames or []
        speech_ms, silence_ms = self._speech_ms(), self._ms(self._silence)
        level, floor = self.level_dbfs, self._floor_at_start
        self.last_level, self.last_floor = level, floor
        min_speech, announced = self.min_speech, self._confirmed
        self._frames, self._samples, self._silence, self._paused = None, 0, 0, False
        self._levels, self._floor_at_start, self._confirmed = [], None, False
        for frame in frames[-max(1, round(self._pre_roll_ms * self.sample_rate / 1000 / FRAME_SAMPLES)) :]:
            self._remember(frame)  # the trailing silence is the next utterance's pre-roll
        if not announced or speech_ms < min_speech:
            return SpeechDiscarded(speech_ms, announced)
        return SpeechEnded(Utterance(np.concatenate(frames), speech_ms, silence_ms, level, floor))

    def pre_roll(self) -> list[np.ndarray]:
        """The audio kept from before now (PRE_ROLL_MS), for an utterance that starts without the VAD (hold-to-talk)."""
        return list(self._pre)

    def reset(self) -> None:
        self._pre.clear()
        self._pre_samples = 0
        self._frames, self._samples, self._silence, self._paused = None, 0, 0, False
        self._levels, self._floor_at_start, self._confirmed = [], None, False


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
    cap_passed: bool = False  # the extended deadline (acknowledgements, or a transcription still running) has passed
    transcribing: bool = False  # a transcription of this speech is running (its result may come after the deadline)
    stop_words: bool = False  # the transcript is a stop phrase ("Stop.", "बस", "ruko"): stop at once, partial or not


Verdict = Literal["stop", "resume"]


def barge_in_verdict(evidence: BargeInEvidence, *, min_speech_ms: float, is_backchannel: bool | None) -> Verdict | None:
    """Duck, then decide (§3.3 c). ``is_backchannel`` is the transcript's classification (None without a transcript;
    hums, fillers, noise and empty transcripts are backchannels).

    0. a transcript that is a stop phrase ("Stop.", "बस", "ruko")     → stop, at once, partial or not
    1. a transcript that isn't a backchannel                          → stop (as soon as it is known), except a
       partial of fewer than 2 real words of speech that has already stopped: Whisper often mishears a short
       snapshot ("M M" for "mm-hmm"), so that waits for the deadline or the final transcript
    2. the utterance ended as a backchannel, noise or too short        → resume
    3. before the deadline                                             → wait (None)
    4. at the deadline: speech shorter than min_speech_ms              → resume
                        a short burst that has stopped                 → resume; if its final transcript turns out not
                                                                         to be a backchannel, the session still stops
                                                                         the answer when the utterance ends
                        a transcription still running (B3: under      → wait for it, until the extended deadline;
                        load the snapshot of "Yeah, right" often        at the cap the rules below decide (no
                        comes after the deadline)                       transcript, still talking → stop)
                        still talking, no transcript and none coming, → stop
                        or not a backchannel
                        still talking, but only acknowledgements so    → wait until the extended deadline (the session
                        far ("Yeah…" of "Yeah, right")                   transcribes again meanwhile), then resume:
                                                                         only real words stop the answer
    """
    if evidence.transcript is not None and evidence.stop_words:
        return "stop"  # "Stop." as a partial: never "resume" first (the volume came back for ~0.3 s, last round)
    sure = evidence.speaking or evidence.ended or evidence.real_words >= 2
    if evidence.transcript is not None and is_backchannel is False and sure:
        return "stop"
    if evidence.ended:
        return "resume"
    if not evidence.deadline_passed:
        return None
    if evidence.speech_ms < min_speech_ms:
        return "resume"
    if evidence.transcribing and not evidence.cap_passed:
        return None  # a (fuller) transcript is coming: decide on it, at the latest at the cap
    if not evidence.speaking:
        return "resume"
    if evidence.transcript is not None and is_backchannel:
        return "resume" if evidence.cap_passed else None
    return "stop"
