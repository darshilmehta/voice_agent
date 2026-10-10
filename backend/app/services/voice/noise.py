"""Noise robustness for the voice session (docs/DESIGN.md §3.10 "Noisy rooms"): the room's noise floor, the speech
gate it implies, and whether an utterance was addressed to the agent at all.

    every 32 ms frame → its level (dBFS) → NoiseFloor: a low percentile of the last few seconds of levels (steady noise
                        and babble alike: speech has gaps, so the percentile stays near what is always there)
    NoiseGate          → a turn opens only on a frame the VAD calls speech *and* that is start_snr_db above the floor;
                         the louder the floor (quiet_floor_dbfs → loud_floor_dbfs), the higher the VAD threshold and the
                         longer the speech a turn needs (noisy_threshold, noisy_min_speech_ms); frames within end_snr_db
                         of the floor count as silence, so babble alone doesn't hold a turn open; and the turn is only
                         announced once its level is near the user's (``near_field``: within far_field_hard_db)
    addressed()        → answer, say again, or drop silently: speech far below the user's own voice (far field: the
                         TV, the next table), or speech that is weak against the floor *and* unsure or a fragment with
                         no content words, was not said to the agent; "say again" is only for near-field speech that
                         was garbled
    UserLevel          → the user's own speech level, learnt from the turns that were answered (and every hold-to-talk
                         turn); before the first one, ``assumed_user_dbfs`` with assumed_margin_db more room

An utterance's level is the LEVEL_PERCENTILE-th percentile of its speech frames' levels: its loud part, so speech that
began in background talk and went on as the user's question counts as the user's.

A denoiser in front (the browser's RNNoise) removes most of the steady noise between words, so after it the floor is
low and babble stands well above it: against babble and the TV it is the level relative to the user's that tells.

Levels are 10·log10(mean square) of the PCM in [-1, 1] (a full-scale sine is -3 dBFS); the browser computes them the
same way (frontend/lib/voice/noise.ts), so one set of thresholds serves both sides. Everything here is pure and tested
without audio models.
"""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from .speech_text import is_acknowledgement, is_filler, normalize_utterance

DB_MIN = -100.0  # digital silence
FRAME_MS = 32.0  # 512 samples at 16 kHz
MIN_FLOOR_FRAMES = 31  # ~1 s of audio before the floor is measured rather than assumed
QUIET_FLOOR_DBFS = -70.0  # the floor assumed before then (a quiet room)
_RECOMPUTE_EVERY = 4  # frames between two percentile computations (the floor moves over seconds)
LEVEL_PERCENTILE = 80  # an utterance's level: this percentile of its speech frames' levels


class NoiseSettings(Protocol):
    """``voice.noise`` (settings.NoiseSection), or anything with the same fields (tests, the eval script)."""

    adaptive_gating: bool
    floor_window_ms: int
    floor_percentile: float
    start_snr_db: float
    end_snr_db: float
    quiet_floor_dbfs: float
    loud_floor_dbfs: float
    noisy_threshold: float
    noisy_min_speech_ms: int
    drop_background_speech: bool
    assumed_user_dbfs: float
    assumed_margin_db: float
    far_field_db: float
    far_field_hard_db: float
    close_db: float
    min_snr_db: float
    unsure_avg_logprob: float
    unsure_no_speech_prob: float
    short_fragment_words: int


def frame_dbfs(frame: np.ndarray) -> float:
    """A frame's level: 10·log10 of its mean square, at least DB_MIN."""
    if frame.size == 0:
        return DB_MIN
    power = float(np.mean(np.square(frame, dtype=np.float64)))
    return max(DB_MIN, 10 * math.log10(power)) if power > 0 else DB_MIN


def utterance_level(levels: list[float]) -> float | None:
    """An utterance's speech level from its speech frames' levels (None without any)."""
    return float(np.percentile(levels, LEVEL_PERCENTILE)) if levels else None


class NoiseFloor:
    """A rolling low percentile of frame levels: what the room sounds like when nobody near the mic is talking."""

    def __init__(self, *, window_ms: float, percentile: float, frame_ms: float = FRAME_MS) -> None:
        self.percentile = percentile
        self._levels: deque[float] = deque(maxlen=max(MIN_FLOOR_FRAMES, round(window_ms / frame_ms)))
        self._since = 0
        self._value = QUIET_FLOOR_DBFS

    def push(self, dbfs: float) -> None:
        self._levels.append(dbfs)
        self._since += 1
        if len(self._levels) >= MIN_FLOOR_FRAMES and (
            self._since >= _RECOMPUTE_EVERY or len(self._levels) == MIN_FLOOR_FRAMES
        ):
            self._since = 0
            self._value = float(np.percentile(np.fromiter(self._levels, dtype=np.float64), self.percentile))

    @property
    def dbfs(self) -> float:
        return self._value


# ------------------------------------------------------------------ the user's own level


class UserLevel:
    """The user's speech level (dBFS), from turns known to be theirs. It rises quickly and falls slowly: the user is
    the loudest talker near the mic, and one quiet turn (or background speech that got through) shouldn't make every
    later background voice look like theirs."""

    RISE = 0.5
    FALL = 0.15

    def __init__(self) -> None:
        self.dbfs: float | None = None
        self.turns = 0

    def update(self, level_dbfs: float | None) -> None:
        if level_dbfs is None or level_dbfs <= DB_MIN:
            return
        self.turns += 1
        if self.dbfs is None:
            self.dbfs = level_dbfs
            return
        rate = self.RISE if level_dbfs > self.dbfs else self.FALL
        self.dbfs += (level_dbfs - self.dbfs) * rate


class NoiseGate:
    """One session's view of the room: its noise floor and the speech thresholds that follow from it."""

    def __init__(
        self, cfg: NoiseSettings, *, threshold: float, min_speech_ms: float, user: UserLevel | None = None
    ) -> None:
        self.cfg = cfg
        self.user = user or UserLevel()  # the session's: updated as the user's turns are answered
        self.base_threshold = threshold
        self.base_min_speech_ms = min_speech_ms
        self.floor = NoiseFloor(window_ms=cfg.floor_window_ms, percentile=cfg.floor_percentile)

    def observe(self, dbfs: float) -> None:
        self.floor.push(dbfs)

    @property
    def floor_dbfs(self) -> float:
        return self.floor.dbfs

    @property
    def noisiness(self) -> float:
        """0 in a quiet room (floor at or below quiet_floor_dbfs), 1 in a loud one (at or above loud_floor_dbfs)."""
        lo, hi = self.cfg.quiet_floor_dbfs, self.cfg.loud_floor_dbfs
        return min(1.0, max(0.0, (self.floor_dbfs - lo) / (hi - lo)))

    @property
    def threshold(self) -> float:
        """The VAD probability a turn needs to open: the configured one, raised toward noisy_threshold."""
        top = max(self.cfg.noisy_threshold, self.base_threshold)
        return self.base_threshold + (top - self.base_threshold) * self.noisiness

    @property
    def min_speech_ms(self) -> float:
        top = max(float(self.cfg.noisy_min_speech_ms), self.base_min_speech_ms)
        return self.base_min_speech_ms + (top - self.base_min_speech_ms) * self.noisiness

    def opens(self, probability: float, dbfs: float) -> bool:
        """This frame may start a turn: speech by the VAD (at the adapted threshold) and loud enough above the floor."""
        return probability >= self.threshold and dbfs >= self.floor_dbfs + self.cfg.start_snr_db

    def audible(self, dbfs: float) -> bool:
        """Above the floor by end_snr_db: a frame at the floor is silence, whatever the VAD hears in it (babble)."""
        return dbfs >= self.floor_dbfs + self.cfg.end_snr_db

    def near_field(self, level_dbfs: float | None) -> bool:
        """Loud enough to be the user: within far_field_hard_db of their level (and assumed_margin_db more while
        it is only assumed). Unknown levels pass."""
        return level_dbfs is None or level_dbfs >= near_field_min(self.cfg, self.user.dbfs)


def near_field_min(cfg: NoiseSettings, user_dbfs: float | None) -> float:
    """The quietest level that can still be the user (below it: far field, dropped whatever was said)."""
    if user_dbfs is not None:
        return user_dbfs - cfg.far_field_hard_db
    return cfg.assumed_user_dbfs - cfg.far_field_hard_db - cfg.assumed_margin_db


# ------------------------------------------------------------------ was it said to the agent?

Addressing = Literal["answer", "say_again", "drop"]

# Words that carry no request on their own (English, romanized Hindi, Devanagari): a fragment of only these ("and then
# he", "Thank you.", "haan to", "है ना") is background chatter or Whisper's filler for noise, not a question.
_FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "so", "of", "to", "in", "on", "at", "for", "with", "from", "by", "as", "if",
    "is", "are", "was", "were", "be", "been", "am", "it", "its", "it's", "this", "that", "these", "those", "there",
    "here", "i", "i'm", "im", "me", "my", "you", "your", "he", "she", "him", "her", "his", "we", "us", "our", "they",
    "them", "their", "do", "does", "did", "have", "has", "had", "not", "no", "yes", "yeah", "okay", "ok", "oh", "well",
    "just", "like", "really", "very", "too", "also", "then", "now", "all", "some", "any", "get", "got", "go", "going",
    "know", "think", "say", "said", "see", "let", "let's", "lets", "can", "could", "would", "should", "will", "shall",
    "may", "might", "must", "one", "thank", "thanks", "bye", "goodbye", "hello", "hi", "hey", "please", "right", "sure",
    "gonna", "wanna", "don't", "dont", "that's", "thats", "there's", "what", "who", "out", "up", "down", "off", "more",
    "back", "come", "came", "nope", "sir", "man", "guys", "huh", "wow", "uh", "um", "ah", "hmm",
    "hai", "hain", "tha", "thi", "ka", "ki", "ke", "ko", "se", "mein", "par", "aur", "ya", "toh",
    "bhi", "na", "nahi", "nahin", "haan", "ji", "kya", "yeh", "ye", "woh", "vo", "wo", "main", "mai", "hum", "tum",
    "aap", "ek", "acha", "achha", "accha", "theek", "thik", "ho", "hota", "raha", "rahi", "rahe", "kar", "karo",
    "diya", "gaya", "arre", "arey", "yaar", "bas",
    "है", "हैं", "था", "थी", "थे", "का", "की", "के", "को", "से", "में", "पर", "और", "या", "तो", "भी", "ही", "न", "ना",
    "नहीं", "हाँ", "हां", "जी", "क्या", "यह", "ये", "वह", "वो", "मैं", "हम", "तुम", "आप", "इस", "उस", "एक", "अच्छा",
    "ठीक", "हो", "रहा", "रही", "रहे", "कर", "करो", "दिया", "गया", "अरे", "यार",
}  # fmt: skip
# Said to stop the agent, however short: never a fragment to drop as chatter.
_STOP_CUES = {"stop", "wait", "pause", "hold", "ruko", "ruk", "रुको", "रुकिए", "रुक", "बस"}
_LETTERS = re.compile(r"[^\W_]", re.UNICODE)


def content_words(text: str) -> int:
    """Words that could carry a request: not function words, hums or acknowledgements; at least three letters (two
    in Devanagari, whose syllables are fewer letters), or a number."""
    count = 0
    for word in normalize_utterance(text).split():
        if word in _FUNCTION_WORDS or is_filler(word) or is_acknowledgement(word):
            continue
        letters = len(_LETTERS.findall(word))
        devanagari = any("ऀ" <= ch <= "ॿ" for ch in word)
        if any(ch.isdigit() for ch in word) or letters >= (2 if devanagari else 3):
            count += 1
    return count


def is_fragment(text: str, max_words: int) -> bool:
    """A short utterance with nothing in it to act on ("and then", "Thank you.", "है ना"); a stop cue is never one."""
    words = normalize_utterance(text).split()
    if not words or len(words) > max_words:
        return False
    if any(w in _STOP_CUES for w in words):
        return False
    return content_words(text) == 0


@dataclass(frozen=True, slots=True)
class SpeechEvidence:
    """What the session knows about one utterance when deciding whether it was said to the agent."""

    text: str
    level_dbfs: float | None  # the utterance's speech level (median of its speech frames), None if unknown
    floor_dbfs: float | None  # the room's noise floor when it began
    user_dbfs: float | None  # the user's established level, None before their first answered turn
    garbled: bool = False  # speech_text.transcript_garbled
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


@dataclass(frozen=True, slots=True)
class AddressDecision:
    verdict: Addressing
    reason: str  # for the log: why (or "near field")
    snr_db: float | None = None
    deficit_db: float | None = None  # how far below the user's level (or the assumed one)


def addressed(e: SpeechEvidence, cfg: NoiseSettings) -> AddressDecision:
    """Answer it, ask to say it again, or drop it silently (it wasn't said to the agent):

        far below the user's level by far_field_hard_db                           → drop (the TV, the next table)
        below it by far_field_db, and unsure, a fragment, or weak vs floor        → drop
        weak against the floor (min_snr_db), not close to the user's level, and
        unsure or a fragment                                                      → drop (babble, the next table)
        garbled                                                                   → say again (near field only)
        otherwise                                                                 → answer

    "Unsure": garbled, Whisper's avg_logprob below unsure_avg_logprob or no_speech_prob above unsure_no_speech_prob.
    Before the user's first answered turn their level is assumed (assumed_user_dbfs, with assumed_margin_db added to
    both far-field distances); "close to it" needs a measured one. With ``drop_background_speech`` off, only garbled
    → say again (the behaviour before this was added)."""
    if not cfg.drop_background_speech:
        return AddressDecision("say_again" if e.garbled else "answer", "background check off")
    if not normalize_utterance(e.text):
        return AddressDecision("drop", "no words")
    snr = deficit = None
    far = very_far = close = weak = False
    if e.level_dbfs is not None and e.level_dbfs > DB_MIN:
        known = e.user_dbfs is not None
        reference = e.user_dbfs if e.user_dbfs is not None else cfg.assumed_user_dbfs
        margin = 0.0 if known else cfg.assumed_margin_db
        deficit = reference - e.level_dbfs
        far = deficit >= cfg.far_field_db + margin
        very_far = deficit >= cfg.far_field_hard_db + margin
        close = e.user_dbfs is not None and deficit <= cfg.close_db
        if e.floor_dbfs is not None:
            snr = e.level_dbfs - e.floor_dbfs
            weak = snr < cfg.min_snr_db
    unsure = (
        e.garbled
        or (e.avg_logprob is not None and e.avg_logprob < cfg.unsure_avg_logprob)
        or (e.no_speech_prob is not None and e.no_speech_prob > cfg.unsure_no_speech_prob)
    )
    fragment = is_fragment(e.text, cfg.short_fragment_words)

    def decision(verdict: Addressing, reason: str) -> AddressDecision:
        return AddressDecision(verdict, reason, snr, deficit)

    if very_far:
        return decision("drop", "far field")
    if far and (unsure or fragment or weak):
        why = "unsure" if unsure else "fragment" if fragment else "weak"
        return decision("drop", f"far field, {why}")
    if weak and not close and (unsure or fragment):
        return decision("drop", f"weak against the noise, {'unsure' if unsure else 'fragment'}")
    if e.garbled:
        return decision("say_again", "garbled near field")
    return decision("answer", "near field")
