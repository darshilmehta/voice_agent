"""A live voice conversation in one chat: the state machine behind ``WS /ws/chats/{chat_id}/voice`` (docs/DESIGN.md
§3.3, §3.5, §3.9, §3.10).

    LISTENING ──end of turn (server VAD)──► THINKING ──first audio chunk──► SPEAKING ──playback done──► LISTENING
        ▲                                     │   ▲                            │
        │                                     │   └── the next turn starts ◄───┤ barge-in confirmed / "stop":
        └──────────── INTERRUPTED ◄───────────┴────────────────────────────────┘ answer cut, what was heard saved

- Audio in: PCM16 16 kHz → Silero (this session's state) → ``Endpointer``. At half the end-of-turn silence the
  utterance is transcribed speculatively; at the end of the turn that transcript is reused unless speech resumed (a
  stale speculative job is cancelled, so it never queues in front of the next one).
- A user turn is ``ChatTurnService`` with ``modality="voice"``, ``length="short"``: the same pipeline as text chat.
  The session saves the user message first (so a stop can't lose it), then runs the answer as a task. Deltas go to
  the client and through ``SpeechChunker`` to TTS; each chunk is announced (``audio_chunk``), then its binary frames
  follow (one send-lock acquisition per frame, so control messages aren't held up). ``agent_message`` is sent once the
  last frame is out.
- Barge-in, duck-then-decide: the client ducks and sends ``barge_in_start``; the server decides within
  ``voice.barge_in.decision_timeout_ms`` from its own VAD and transcripts of the new speech (``barge_in_verdict``).
  A transcription still running at that deadline is waited for, up to the acknowledgement cap (~1.4 s, B3).
- Speech that goes on while the agent answers with no decision pending (after a ``resume``: "No… wait, I meant…";
  or the client's VAD missed it) is transcribed every ``WATCH_STEP_MS`` of new speech; real words that aren't an
  acknowledgement stop the answer at once (``barge_in: stop`` without ``barge_in_start``), not when the sentence
  ends (B3).
  On "stop" the turn is muted and its task cancelled at once, then the turn is *settled* by its own task: the
  decision is sent, the stopped answer saved with ``heard_text`` (an empty one if nothing had been generated) and sent
  as ``agent_message``. The next turn waits for pending settles, so the cut answer always precedes the next
  ``user_message``. An utterance that ends while the agent is answering and isn't a backchannel stops the answer too,
  even without ``barge_in_start``.
- An utterance that is ignored (too short, a backchannel, hums, noise, failed STT) still leaves the client in a known
  state: ``user_speech end``, ``barge_in resume`` if a decision was pending, and the current ``state`` again.
- Failures are reported as ``error {stage}`` and the session goes on; repeated input errors at most every few seconds.
- Noisy rooms (§3.10): the endpointer gates turns on the room's noise floor (``noise.NoiseGate``), and every utterance
  is checked for whether it was said to the agent at all (``noise.addressed``): background speech (the TV, the next
  table) is dropped silently, before it can start a turn or cut an answer; only near-field speech that was garbled is
  asked again. In hold-to-talk mode (``input_mode: "ptt"``) the VAD opens no turn: a turn is the audio between
  ``ptt down`` and ``ptt up``, and "down" while the agent answers stops it at once.
- Live data (§3.7): the turn's ``tool`` events go to the client as ``tool`` messages (the "searching the web"
  badge). On ``tool start`` the pre-synthesized filler ("Let me look that up.", ``voice/fillers.py``) is sent at
  once, with its frames, as the turn's first ``audio_chunk`` (``filler: true``), before the turn's next message;
  the answer's chunks follow, and ``state: speaking`` comes with the first of them. The filler isn't part of what
  was heard of the answer. A barge-in or stop during the search cancels the turn, and the search with it. The
  answer's last sentence is spoken as soon as its text is complete (``AnswerStop.on_answered``), not after a
  continuation; speech after the complete answer has been played, while only a continuation was still to come,
  isn't a barge-in: the turn just ends (the answer saved complete) and the speech is the next turn.
- The canvas (§12.1): the answer's visual comes from the turn (``run(on_visual=…)``) as ``visual`` / ``canvas`` with
  the turn's id. Its draft is ready while the answer is still being written: it is held until the answer's first
  audio and sent right after it (never before); a planner's refinement replaces it in place, before or after
  ``agent_message``. Ready while the answer is still being heard (and, for a draft the planner is checking, kept by
  it), it is followed by the tail ("It's on screen now.", ``audio_chunk {tail: true}``, not part of the answer). A turn
  cut before its first audio withdraws its draft; ``stop`` and the session ending cancel a refinement (the draft
  stays); the next question cancels it in the turn service.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections.abc import Awaitable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from ...db.types import new_id
from ...domain.canvas import CanvasEvent, VisualEvent
from ...domain.projects import Chat, Message
from ...providers.registry import Container
from ...providers.speech import SpeechRecognizer, SpeechSynthesizer, Transcript, VoiceActivityDetector
from ...providers.storage import MetadataDB
from ...settings import Language, Settings
from ..base import InvalidInput, NotFound
from ..canvas.conversation import TurnVisual, visual_in_progress
from ..chat_turns import (
    AgentMessageEvent,
    AnswerStop,
    ChatEvent,
    ChatTurnService,
    DeltaEvent,
    ErrorEvent,
    InterruptReason,
    SourcesEvent,
    ToolEvent,
    Turn,
    UserMessageEvent,
    detach,
)
from ..chats import ChatService
from ..language import script_language
from ..messages import MessageService
from ..router import STOP_PHRASES, normalize
from .fillers import VISUAL_TAILS, filler_audio
from .noise import AddressDecision, NoiseGate, SpeechEvidence, UserLevel, addressed, frame_dbfs, utterance_level
from .protocol import (
    CLOSE_CHAT_NOT_FOUND,
    CLOSE_GOING_AWAY,
    CLOSE_INTERNAL_ERROR,
    CLOSE_NORMAL,
    CLOSE_REPLACED,
    INPUT_SAMPLE_RATE,
    MAX_INPUT_FRAME_BYTES,
    AgentState,
    BargeInStart,
    End,
    ErrorStage,
    InputMode,
    Playback,
    PlaybackDone,
    ProtocolError,
    Ptt,
    SetInputMode,
    Start,
    Stop,
    audio_frames,
    dumps,
    parse_client_message,
)
from .speech_text import (
    SpeechChunker,
    SpokenChunk,
    heard_text,
    is_backchannel,
    is_filler,
    normalize_utterance,
    real_words,
    transcript_garbled,
    transcript_unsure,
)
from .turn_taking import (
    FRAME_SAMPLES,
    MAX_UTTERANCE_MS,
    SAMPLE_RATE,
    BargeInEvidence,
    Endpointer,
    SpeechDiscarded,
    SpeechEnded,
    SpeechPaused,
    SpeechResumed,
    SpeechStarted,
    Utterance,
    VADEvent,
    Verdict,
    barge_in_verdict,
)
from .vocabulary import SpeechHints, SpeechVocabulary, speech_vocabulary

if TYPE_CHECKING:
    from ..canvas.service import CanvasService

log = logging.getLogger(__name__)

PLAYBACK_GRACE_S = 2.0  # a turn whose client never says playback_done ends this long after its audio should have
CLOSE_WAIT_S = 10.0  # how long a replaced session may take to save its state and close
ERROR_REPEAT_S = 5.0  # the same kind of input error (bad frames, VAD, bad control messages) at most this often
BARGE_IN_MATCH_S = 0.5  # a barge_in_start belongs to an utterance that started at most this long before it
# Speech still going on at decision_timeout_ms but only acknowledgements so far ("Yeah…" of "Yeah, right"): the decision
# waits up to this many times decision_timeout_ms (700 → 1400 ms) for more words; only real words stop the answer.
ACKNOWLEDGEMENT_GRACE = 2
# Speech going on while the agent answers, with no barge-in decision pending: transcribed again after this much more
# speech (the first time at decision_timeout_ms of speech, or this long after a "resume"), to stop the answer as soon
# as it has real words instead of when the sentence ends (B3).
WATCH_STEP_MS = 400
# The visual's spoken tail (§12.1) is sent after agent_message only while the answer still has this much left to play:
# the client plays it as part of the turn, before it reports playback_done.
TAIL_MARGIN_S = 0.3
# An answer that already said the chart is there ("The chart shows …", quality round item 1) gets no spoken tail.
_NAMES_THE_CHART = re.compile(r"\b(?:chart|graph)s?\b|चार्ट|ग्राफ़|ग्राफ", re.IGNORECASE)


class Transport(Protocol):
    """The connection, as the session sees it (the WebSocket endpoint adapts Starlette's WebSocket to this)."""

    async def receive(self) -> bytes | str | None:
        """The next frame: bytes (audio) or str (JSON); None once the client has disconnected."""
        ...

    async def send_text(self, text: str) -> None: ...

    async def send_bytes(self, data: bytes) -> None: ...

    async def close(self, code: int, reason: str = "") -> None: ...


def _describe(e: BaseException) -> str:
    text = str(e)
    return f"{type(e).__name__}: {text}" if text else type(e).__name__


def _message_json(message: Message) -> dict[str, Any]:
    return message.model_dump(mode="json")


SpeakItem = str | None  # text to synthesize, or the end of the answer


@dataclass(eq=False)
class AgentTurn:
    """One answer: generation, speech and playback of the reply to one user utterance."""

    id: int
    turn: Turn
    stop: AnswerStop = field(default_factory=AnswerStop)
    task: asyncio.Task[None] | None = None
    chunks: list[SpokenChunk] = field(default_factory=list)  # announced to the client, in order
    play_starts: list[float] = field(default_factory=list)  # when the client should start playing each chunk
    play_end: float = 0.0  # perf_counter when the client should have played everything sent so far
    sent_ms: float = 0.0  # audio sent so far
    message: Message | None = None  # the saved, complete answer
    audio_done: bool = False  # generation finished and every chunk was sent
    errored: bool = False  # the pipeline reported an error: the cut-off fragment is not spoken
    cut: bool = False  # stopped or barged in: nothing more of this turn (sources, deltas, audio) is sent
    interrupted: bool = False  # the interruption has been handled (or is being handled)
    user_sent: bool = False  # user_message sent to the client
    turn_sent: bool = False  # turn sent to the client
    tts_failed: bool = False
    first_audio_at: float | None = None  # the answer's first audio (not the filler's)
    queue: asyncio.Queue[SpeakItem] | None = None  # text waiting for TTS
    voice_language: Language | None = None  # the voice speaking this answer: its text's script, once it tells (B5)
    tts_busy: bool = False  # the speaker is synthesizing or sending a chunk
    answer_queued: bool = False  # the answer's text is complete and all of it is queued for speech
    message_sent: bool = False  # agent_message was sent
    reported_ms: float | None = None  # last playback progress from the client
    reported_at: float = 0.0
    barge_in_ms: float | None = None  # played_ms of the latest barge_in_start
    barge_in_at: float | None = None  # when it arrived (perf_counter)
    playback_timer: asyncio.TimerHandle | None = None
    marks: dict[str, float] = field(default_factory=dict)  # perf_counter timestamps for the latency log
    answer_ms: float | None = None  # the answer's audio, once all of it is sent (the visual's tail comes after)
    tail_wanted: bool = False  # the visual is ready while the answer is still being spoken: say so after it
    tail_sent: bool = False
    held_visual: list[VisualEvent | CanvasEvent] = field(default_factory=list)  # ready before the first audio
    visual_released: bool = False  # the answer's first audio is out: its visual's events go out as they come


def is_stop_phrase(text: str) -> bool:
    """A transcript that only asks the agent to stop ("Stop.", "stop it", "bas karo", "रुको")."""
    return normalize(text) in STOP_PHRASES


@dataclass(eq=False)
class PendingBargeIn:
    agent: AgentTurn
    played_ms: float
    deadline: asyncio.Task[None] | None = None
    stt: asyncio.Task[None] | None = None
    transcript: str | None = None
    backchannel: bool | None = None
    real_words: int = 0
    stop_words: bool = False  # the transcript is a stop phrase
    deadline_passed: bool = False
    cap_passed: bool = False
    transcribing: int = 0  # transcriptions of the interrupting speech still running
    # The interrupting utterance has ended and gone to the worker, whose final transcript decides if nothing has yet:
    # its speech (the endpointer reads 0 once the utterance is closed) and no "resume" at the deadline or the cap.
    ended_speech_ms: float | None = None


@dataclass(eq=False)
class SpeechWatch:
    """Speech over an answer that nobody is deciding about (after a "resume", or without ``barge_in_start``)."""

    agent: AgentTurn
    utterance: int  # the endpointer's utterance number
    checked_ms: float  # speech_ms at the last transcription
    stt: asyncio.Task[None] | None = None


@dataclass(eq=False)
class _EndedUtterance:
    utterance: Utterance
    speculative: asyncio.Task[Transcript | None] | None
    ended_at: float  # perf_counter at the end of turn
    ptt: bool = False  # held to talk: said to the agent by definition (no background check)

    @property
    def started_at(self) -> float:
        """When the speech began (audio arrives in real time, so audio time is wall time)."""
        return self.ended_at - (self.utterance.speech_ms + self.utterance.silence_ms) / 1000


@dataclass(eq=False)
class _PttCapture:
    """The audio of one hold-to-talk press, from the pre-roll before "down" until "up"."""

    frames: list[np.ndarray]
    floor_dbfs: float | None
    samples: int = 0
    speech_samples: int = 0  # in frames the VAD calls speech
    levels: list[float] = field(default_factory=list)  # their levels

    def add(self, frame: np.ndarray, speech: bool) -> None:
        self.frames.append(frame)
        self.samples += len(frame)
        if speech:
            self.speech_samples += len(frame)
            self.levels.append(frame_dbfs(frame))


class VoiceSession:
    def __init__(
        self,
        chat: Chat,
        transport: Transport,
        *,
        turns: ChatTurnService,
        messages: MessageService,
        vad: VoiceActivityDetector,
        stt: SpeechRecognizer,
        tts: SpeechSynthesizer,
        settings: Settings,
        vocabulary: SpeechVocabulary | None = None,
    ) -> None:
        self.id = new_id("vsn")
        self.chat = chat
        self.transport = transport
        self.turns = turns
        self.messages = messages
        self.stt = stt
        self.tts = tts
        self.settings = settings
        self.vad_settings = settings.vad
        self.barge_in = settings.voice.barge_in
        self._vad_stream = vad.open_stream()
        self.noise = settings.voice.noise
        self._user_level = UserLevel()  # the user's own speech level, from turns known to be theirs
        self._gate = NoiseGate(
            self.noise,
            threshold=settings.vad.threshold,
            min_speech_ms=settings.vad.min_speech_ms,
            user=self._user_level,
        )
        self._endpointer = Endpointer(
            threshold=settings.vad.threshold,
            min_speech_ms=settings.vad.min_speech_ms,
            end_of_turn_ms=settings.vad.end_of_turn_ms,
            gate=self._gate,
        )
        self._input_mode: InputMode = "vad"
        self._ptt: _PttCapture | None = None  # the talk button is down
        self._language: Language | None = None
        self._started = False
        self._rest = np.zeros(0, dtype=np.float32)
        self._turn: AgentTurn | None = None
        self._turn_ids = 0
        self._turn_lock = asyncio.Lock()  # one settle at a time
        self._settling: set[asyncio.Task[None]] = set()  # interrupted turns being saved; the next turn waits for them
        self._pending: PendingBargeIn | None = None
        self._watch: SpeechWatch | None = None
        self._utterance_no = 0  # counts the endpointer's utterances
        self._speculative: asyncio.Task[Transcript | None] | None = None
        self._utterances: asyncio.Queue[_EndedUtterance] = asyncio.Queue()
        self._processing = False  # the worker is transcribing or starting a turn for an utterance
        self._stop_at: float | None = None  # the last "stop": utterances that ended before it are not answered
        self._send_lock = asyncio.Lock()
        self._open = True  # the transport can still be written to
        self._close_requested = asyncio.Event()
        self._close_code = CLOSE_NORMAL
        self._close_reason = ""
        self._finished = asyncio.Event()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._state: AgentState | None = None
        self._errors_sent: dict[str, float] = {}
        self._tails: set[Language] = set()  # languages whose visual tail is synthesized (or being)
        # The chat's names and terms for the recognizer (§9.3): refreshed in the background, never awaited by STT.
        self._vocabulary = vocabulary
        self._hints: SpeechHints | None = vocabulary.cached(chat.id) if vocabulary is not None else None
        self._hints_task: asyncio.Task[None] | None = None

    @classmethod
    def from_container(
        cls,
        chat: Chat,
        transport: Transport,
        container: Container,
        *,
        canvas: CanvasService | None = None,
        vocabulary: SpeechVocabulary | None = None,
    ) -> VoiceSession:
        db, vad, stt, tts = (container[c] for c in ("metadata_db", "vad", "stt", "tts"))
        if not (
            isinstance(db, MetadataDB)
            and isinstance(vad, VoiceActivityDetector)
            and isinstance(stt, SpeechRecognizer)
            and isinstance(tts, SpeechSynthesizer)
        ):
            raise TypeError("the voice session needs metadata_db, vad, stt and tts providers")
        return cls(
            chat,
            transport,
            turns=ChatTurnService.from_container(container, canvas=canvas),
            messages=MessageService(db),
            vad=vad,
            stt=stt,
            tts=tts,
            settings=container.settings,
            vocabulary=vocabulary,
        )

    # -------------------------------------------------------------- lifecycle

    async def run(self) -> None:
        """Serve the connection until the client ends it, disconnects, or ``close()`` is called.

        The clean-up (cut and save the current answer, close the connection) runs as its own task, so it completes
        even when the server cancels this one (anyio cancel scopes re-cancel at every await)."""
        receiver = asyncio.create_task(self._receive_loop(), name=f"{self.id}-receive")
        worker = asyncio.create_task(self._utterance_worker(), name=f"{self.id}-utterances")
        closer = asyncio.create_task(self._close_requested.wait())
        loops = (receiver, worker, closer)
        try:
            done, _ = await asyncio.wait(loops, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task is not closer and not task.cancelled() and task.exception() is not None:
                    log.error("voice session %s failed", self.id, exc_info=task.exception())
                    self._close_code, self._close_reason = CLOSE_INTERNAL_ERROR, "internal error"
        finally:
            await asyncio.shield(detach(self._shutdown(loops)))

    def close(self, code: int = CLOSE_NORMAL, reason: str = "") -> None:
        """Ask the session to end: the current answer is cut and saved, then the connection closes with ``code``."""
        if not self._close_requested.is_set():
            self._close_code, self._close_reason = code, reason
            self._close_requested.set()

    async def wait_closed(self, timeout: float = CLOSE_WAIT_S) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._finished.wait(), timeout)

    async def _shutdown(self, loops: tuple[asyncio.Task[Any], ...]) -> None:
        try:
            for task in loops:
                task.cancel()
            await asyncio.wait(loops)
            pending, self._pending = self._pending, None
            if pending is not None and pending.deadline is not None:
                pending.deadline.cancel()
            self._cancel_visual()  # nobody is left to see it
            if self._turn is not None:
                await self._interrupt(self._turn, "disconnect", None)
            if self._settling:  # interruptions started before (they save what was heard): let them finish
                await asyncio.wait(set(self._settling))
            tasks = list(self._tasks)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.wait(tasks)
            if self._open:
                self._open = False
                with contextlib.suppress(Exception):
                    await self.transport.close(self._close_code, self._close_reason)
            log.info("voice session %s for chat %s closed (%d)", self.id, self.chat.id, self._close_code)
        finally:
            self._finished.set()

    def _spawn(self, coro: Awaitable[Any], name: str | None = None) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        if name:
            task.set_name(name)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("voice session %s: background task failed", self.id, exc_info=task.exception())

    # -------------------------------------------------------------- sending

    async def _send(self, message: dict[str, Any]) -> None:
        async with self._send_lock:
            await self._send_text_unlocked(dumps(message))

    async def _send_text_unlocked(self, text: str) -> None:
        if not self._open:
            return
        try:
            await self.transport.send_text(text)
        except Exception:  # the client is gone: stop the session
            self._open = False
            self._close_requested.set()

    async def _send_bytes_unlocked(self, data: bytes) -> None:
        if not self._open:
            return
        try:
            await self.transport.send_bytes(data)
        except Exception:
            self._open = False
            self._close_requested.set()

    async def _send_for(self, agent: AgentTurn, message: dict[str, Any]) -> None:
        """Send a message of a turn unless the turn has been cut (checked under the send lock, so nothing of a cut
        turn follows the barge_in decision or the reaction to stop)."""
        async with self._send_lock:
            if not agent.cut:
                await self._send_text_unlocked(dumps(message))

    async def _error(self, stage: ErrorStage, detail: str, *, repeat_key: str | None = None) -> None:
        """Report a failure. With ``repeat_key``, errors of that kind are sent at most every ERROR_REPEAT_S (a client
        sending bad frames 30 times a second gets one error, not 30)."""
        if repeat_key is not None:
            now = time.monotonic()
            last = self._errors_sent.get(repeat_key)
            if last is not None and now - last < ERROR_REPEAT_S:
                return
            self._errors_sent[repeat_key] = now
        await self._send({"type": "error", "detail": detail, "stage": stage})

    async def _set_state(self, state: AgentState) -> None:
        if state != self._state:
            self._state = state
            await self._send({"type": "state", "state": state})

    def _current_state(self) -> AgentState:
        agent = self._turn
        if agent is not None:  # the filler alone isn't "speaking": the answer hasn't started
            return "speaking" if any(not c.filler for c in agent.chunks) else "thinking"
        if self._processing or not self._utterances.empty():
            return "thinking"
        return "listening"

    async def _resend_state(self) -> None:
        """Send the current state again, even if unchanged (after an utterance that was ignored)."""
        self._state = None
        await self._set_state(self._current_state())

    # -------------------------------------------------------------- receiving

    async def _receive_loop(self) -> None:
        while True:
            data = await self.transport.receive()
            if data is None:  # disconnected
                self._open = False
                return
            if isinstance(data, bytes):
                await self._on_audio(data)
                continue
            try:
                message = parse_client_message(data)
            except ProtocolError as e:
                await self._error("audio", str(e), repeat_key="control")
                continue
            if isinstance(message, End):
                return
            await self._on_control(message)

    async def _on_control(
        self, message: Start | BargeInStart | Playback | PlaybackDone | Stop | SetInputMode | Ptt
    ) -> None:
        match message:
            case Start(language=language, input_mode=mode):
                self._language, self._started = language, True
                await self._set_input_mode(mode)
                await self._send(
                    {
                        "type": "ready",
                        "session_id": self.id,
                        "input_sample_rate": INPUT_SAMPLE_RATE,
                        "output_sample_rate": self.tts.sample_rate,
                        "language": language,
                    }
                )
                self._refresh_hints()
                await self._resend_state()
            case BargeInStart():
                await self._on_barge_in_start(message)
            case Playback(turn_id=turn_id, played_ms=played_ms):
                agent = self._turn
                if agent is not None and agent.id == turn_id:
                    now = time.perf_counter()
                    agent.reported_ms, agent.reported_at = played_ms, now
                    # The client tells how far it is: the rest can't finish before now + what is left to play.
                    agent.play_end = max(agent.play_end, now + max(0.0, agent.sent_ms - played_ms) / 1000)
            case PlaybackDone(turn_id=turn_id):
                agent = self._turn
                if agent is not None and agent.id == turn_id and agent.audio_done:
                    await self._finish_turn(agent)
            case Stop():
                # Utterances that ended before this are not answered either (one may be in STT while the client
                # already shows "thinking"): they are saved with an empty, stopped answer (§3.10). A visual still
                # being prepared stops too (§12.1).
                self._stop_at = time.perf_counter()
                self._cancel_visual()
                if self._turn is not None:
                    await self._interrupt(self._turn, "stop", None)
                elif self._processing or not self._utterances.empty():
                    await self._set_state("interrupted")
            case SetInputMode(mode=mode):
                await self._set_input_mode(mode)
            case Ptt(state="down"):
                await self._ptt_down()
            case Ptt(state="up"):
                await self._ptt_up()

    def _languages(self) -> list[Language]:
        return [self._language] if self._language else list(self.settings.stt.languages)

    # -------------------------------------------------------------- audio in

    async def _on_audio(self, data: bytes) -> None:
        if not self._started:
            await self._error(
                "audio", 'audio before start is ignored: send {"type": "start"} first', repeat_key="early-audio"
            )
            return
        if len(data) % 2 or len(data) > MAX_INPUT_FRAME_BYTES:
            await self._error(
                "audio",
                f"audio frames must be PCM16 mono 16 kHz, at most 1 s (got {len(data)} bytes)",
                repeat_key="frame",
            )
            return
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        buffer = np.concatenate([self._rest, samples]) if self._rest.size else samples
        usable = len(buffer) // FRAME_SAMPLES * FRAME_SAMPLES
        self._rest = buffer[usable:].copy()
        if not usable:
            return
        frames = buffer[:usable].reshape(-1, FRAME_SAMPLES)
        try:
            probabilities = await self._vad_stream(frames)
        except Exception as e:
            log.exception("voice session %s: VAD failed", self.id)
            await self._error("audio", f"voice activity detection failed: {_describe(e)}", repeat_key="vad")
            return
        for frame, probability in zip(frames, probabilities, strict=True):
            if self._input_mode == "ptt":
                await self._ptt_audio(frame, probability)
                continue
            for event in self._endpointer.push(frame, probability):
                await self._on_vad_event(event)
        if self._input_mode == "vad":
            await self._barge_in_check()

    async def _on_vad_event(self, event: VADEvent) -> None:
        match event:
            case SpeechStarted():
                self._utterance_no += 1
                await self._send({"type": "user_speech", "phase": "start"})
            case SpeechPaused(audio=audio):
                self._drop_speculative()
                self._speculative = self._spawn(self._speculate(audio), f"{self.id}-speculative-stt")
            case SpeechResumed():
                self._drop_speculative()
            case SpeechEnded(utterance=utterance):
                speculative, self._speculative = self._speculative, None  # handed to the worker, still valid
                if self._pending is not None:
                    self._pending.ended_speech_ms = utterance.speech_ms
                await self._send({"type": "user_speech", "phase": "end"})
                if self._turn is None:
                    await self._set_state("thinking")
                self._utterances.put_nowait(_EndedUtterance(utterance, speculative, time.perf_counter()))
            case SpeechDiscarded(announced=announced):
                self._drop_speculative()
                if announced:  # one the gate never announced (far off, or too short) ends as quietly as it began
                    await self._send({"type": "user_speech", "phase": "end"})
                    await self._ignored_utterance()

    def _drop_speculative(self) -> None:
        """Speech resumed (or a new pause): the speculative transcript is stale. Cancelling it also takes a job that
        hasn't started off the STT thread's queue, so it can't delay the next transcription."""
        if self._speculative is not None:
            self._speculative.cancel()
            self._speculative = None

    async def _transcribe(self, audio: np.ndarray, *, report: bool, hinted: bool = True) -> Transcript | None:
        """``hinted``: with the chat's vocabulary (prompt, and near-miss names corrected); the barge-in checks go
        without, they only need to know whether real words were said."""
        hints = self._hints if hinted else None
        try:
            if hints is not None and hints.prompts:
                transcript = await self.stt.transcribe(audio, self._languages(), prompts=hints.prompts)
            else:
                transcript = await self.stt.transcribe(audio, self._languages())
        except Exception as e:
            log.warning("voice session %s: transcription failed: %s", self.id, _describe(e))
            if report:
                await self._error("stt", f"transcription failed: {_describe(e)}")
            return None
        if hints is not None and (text := hints.correct_hindi(hints.correct(transcript.text))) != transcript.text:
            log.info("voice session %s: transcript names corrected: %r → %r", self.id, transcript.text, text)
            transcript = replace(transcript, text=text)
        return transcript

    def _refresh_hints(self) -> None:
        """Bring the chat's speech hints up to date in the background (rebuilt only when its documents changed)."""
        vocabulary = self._vocabulary
        if vocabulary is None or (self._hints_task is not None and not self._hints_task.done()):
            return

        async def refresh() -> None:
            try:
                self._hints = await vocabulary.hints(self.chat)
            except Exception as e:  # transcription goes on with the hints it has (or none)
                log.warning("voice session %s: speech vocabulary unavailable: %s", self.id, _describe(e))

        self._hints_task = self._spawn(refresh(), f"{self.id}-vocabulary")

    async def _speculate(self, audio: np.ndarray) -> Transcript | None:
        """Transcribe the utterance during the end-of-turn silence (§9.4); also evidence for a pending barge-in."""
        transcript = await self._transcribe(audio, report=False)
        if transcript is not None and transcript.text:
            if not self._background(transcript):  # the TV's words never show as the user's
                await self._send({"type": "transcript_partial", "text": transcript.text})
            pending = self._pending
            if pending is not None:  # a fuller transcript than the barge-in snapshot
                self._note_transcript(pending, transcript)
                await self._evaluate(pending)
        return transcript

    # -------------------------------------------------------------- noisy rooms (§3.10)

    def _addressed(self, transcript: Transcript, level: float | None, floor: float | None) -> AddressDecision:
        evidence = SpeechEvidence(
            transcript.text,
            level,
            floor,
            self._user_level.dbfs,
            garbled=transcript_garbled(transcript),
            avg_logprob=transcript.avg_logprob,
            no_speech_prob=transcript.no_speech_prob,
        )
        return addressed(evidence, self.noise)

    def _background(self, transcript: Transcript) -> bool:
        """The speech being transcribed now (the open utterance, or the one that just ended) wasn't said to the agent:
        it neither shows as a partial transcript nor counts as an interruption."""
        ep = self._endpointer
        level, floor = (ep.level_dbfs, ep.floor_at_start) if ep.in_utterance else (ep.last_level, ep.last_floor)
        return self._addressed(transcript, level, floor).verdict == "drop"

    async def _set_input_mode(self, mode: InputMode) -> None:
        """VAD turns or hold-to-talk. Whatever was being heard in the other mode is dropped (as an ignored
        utterance), so every ``user_speech start`` still gets its ``end``."""
        if mode == self._input_mode:
            return
        self._input_mode = mode
        self._drop_speculative()
        self._watch = None
        if mode == "ptt":
            if self._pending is not None:
                await self._decide(self._pending, "resume")  # only a press interrupts now
            was_open = self._endpointer.in_utterance
            self._endpointer.reset()
            if was_open:
                await self._send({"type": "user_speech", "phase": "end"})
                await self._ignored_utterance()
        else:
            capture, self._ptt = self._ptt, None
            self._endpointer.reset()
            if capture is not None:
                await self._send({"type": "user_speech", "phase": "end"})
                await self._ignored_utterance()
        log.info("voice session %s: input mode %s", self.id, mode)

    async def _ptt_down(self) -> None:
        """The talk button went down: a turn starts now (pre-roll included), and an answer in progress stops."""
        if self._input_mode != "ptt":
            await self._set_input_mode("ptt")
        if self._ptt is not None:
            return  # a repeated "down"
        self._ptt = _PttCapture(self._endpointer.pre_roll(), self._gate.floor_dbfs)
        await self._send({"type": "user_speech", "phase": "start"})
        agent = self._turn
        if agent is not None and not agent.interrupted:
            await self._interrupt(agent, "barge_in", None, decision=True)  # pressing to talk interrupts at once

    async def _ptt_audio(self, frame: np.ndarray, probability: float) -> None:
        self._endpointer.observe(frame)  # the floor and the pre-roll follow the room
        capture = self._ptt
        if capture is None:
            return
        capture.add(frame, probability >= self.vad_settings.threshold)
        if capture.samples * 1000 / SAMPLE_RATE >= MAX_UTTERANCE_MS:
            await self._ptt_up()  # held past Whisper's window: this much is a turn; the rest needs a new press

    async def _ptt_up(self) -> None:
        """The talk button came up: what was said is the user's turn (less than min_speech_ms of speech: ignored)."""
        capture, self._ptt = self._ptt, None
        if capture is None:
            return
        await self._send({"type": "user_speech", "phase": "end"})
        speech_ms = capture.speech_samples * 1000 / SAMPLE_RATE
        if speech_ms < self.vad_settings.min_speech_ms:  # a tap, or nothing said
            await self._ignored_utterance()
            return
        level = utterance_level(capture.levels)
        utterance = Utterance(np.concatenate(capture.frames), speech_ms, 0.0, level, capture.floor_dbfs)
        if self._turn is None:
            await self._set_state("thinking")
        self._utterances.put_nowait(_EndedUtterance(utterance, None, time.perf_counter(), ptt=True))

    # -------------------------------------------------------------- utterances → turns

    async def _utterance_worker(self) -> None:
        while True:
            ended = await self._utterances.get()
            self._processing = True
            try:
                await self._handle_utterance(ended)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # e.g. "database is locked": report it and keep the session going
                log.exception("voice session %s: handling an utterance failed", self.id)
                self._processing = False
                await self._error("storage", f"could not handle what was said: {_describe(e)}")
                await self._resend_state()
            finally:
                self._processing = False

    async def _ignored_utterance(self, *, done_processing: bool = False) -> None:
        """An utterance that won't be answered (too short, backchannel, hums, noise, failed STT): resolve a pending
        barge-in as "resume" and send the current state again, so the client knows what happens next.
        ``done_processing``: called by the worker for the utterance it was handling."""
        if done_processing:
            self._processing = False
        if self._pending is not None:
            await self._decide(self._pending, "resume")
        await self._resend_state()

    async def _handle_utterance(self, ended: _EndedUtterance) -> None:
        transcript: Transcript | None = None
        speculative = False
        if ended.speculative is not None:
            await asyncio.wait([ended.speculative])  # never raises; this task's own cancellation still propagates
            if not ended.speculative.cancelled():
                transcript = ended.speculative.result()
            speculative = transcript is not None
        if transcript is None:
            transcript = await self._transcribe(ended.utterance.audio, report=True)
        stt_ms = (time.perf_counter() - ended.ended_at) * 1000
        if transcript is None:  # STT failed (reported)
            await self._ignored_utterance(done_processing=True)
            return

        text = transcript.text.strip()
        utterance = ended.utterance
        decision = None if ended.ptt else self._addressed(transcript, utterance.level_dbfs, utterance.floor_dbfs)
        if decision is not None and decision.verdict == "drop":
            # Not said to the agent (the TV, the next table): no turn, no "say again", and an answer goes on (§3.10).
            log.info(
                "voice session %s: dropped background speech (%s; level %s dBFS, floor %s, user %s): %r",
                self.id,
                decision.reason,
                _db(utterance.level_dbfs),
                _db(utterance.floor_dbfs),
                _db(self._user_level.dbfs),
                text,
            )
            await self._ignored_utterance(done_processing=True)
            return
        if ended.ptt or (decision is not None and decision.verdict == "answer" and not is_filler(text)):
            self._user_level.update(utterance.level_dbfs)  # the user's own voice: what near field sounds like
        agent = self._turn
        if agent is not None and not agent.interrupted:
            # The agent is answering: is this an interruption?
            if not ended.ptt and is_backchannel(text, self.barge_in.backchannel_max_words):
                await self._ignored_utterance(done_processing=True)  # "mm-hmm", "okay", "M M": the agent goes on
                return
            pending = self._pending if self._pending is not None and self._pending.agent is agent else None
            if self._answer_heard(agent, None):
                # The answer is complete and was heard; only its live-data continuation was still to come (§3.7):
                # not a barge-in. The turn ends there, the answer saved complete, and the speech is the next turn.
                await self._interrupt(agent, "barge_in", None, finished=True)
            elif pending is not None:
                await self._decide(pending, "stop")
            else:  # no barge_in_start (or it was resolved already): use its played_ms only if it was this speech
                await self._interrupt(agent, "barge_in", self._barge_in_played(agent, ended), decision=True)
        # Noise and hums ("M M", "hmm"): nothing to answer. Lexical acknowledgements said while the agent is idle
        # ("okay", "yes please", "theek hai") go to the router, which replies briefly (or not at all) and never
        # abstains, and can tell "yes" to the agent's offer from a bare "okay".
        if not normalize_utterance(text) or is_filler(text):
            await self._ignored_utterance(done_processing=True)
            return
        await self._start_turn(transcript, ended, stt_ms, speculative)

    @staticmethod
    def _barge_in_played(agent: AgentTurn, ended: _EndedUtterance) -> float | None:
        if agent.barge_in_at is None or agent.barge_in_at < ended.started_at - BARGE_IN_MATCH_S:
            return None
        return agent.barge_in_ms

    async def _start_turn(
        self, transcript: Transcript, ended: _EndedUtterance, stt_ms: float, speculative: bool
    ) -> None:
        utterance = ended.utterance
        latency = {
            "speech_ms": round(utterance.speech_ms),
            "end_of_turn_ms": round(utterance.silence_ms),
            "stt_ms": round(stt_ms, 1),
            "speculative_stt": speculative,
        }
        while self._settling:  # an interrupted answer is saved and sent before this user message
            await asyncio.wait(set(self._settling))
        try:
            turn = await self.turns.begin(
                self.chat.id,
                transcript.text,
                language=transcript.language,
                modality="voice",
                length="short",
                input_language=transcript.language,
                input_latency=latency,
                garbled=transcript_garbled(transcript),  # asked to say it again, not answered (quality round)
                unsure=transcript_unsure(transcript),  # declined questions are asked again (last round, item 1)
            )
        except NotFound:
            await self._error("storage", "this chat no longer exists")
            self.close(CLOSE_CHAT_NOT_FOUND, "chat not found")
            return
        except InvalidInput as e:
            await self._error("stt", f"transcript not usable: {e}")
            await self._ignored_utterance(done_processing=True)
            return
        try:
            user = await self.turns.save_user_message(turn)  # before the task: a stop can't lose it
        except Exception as e:
            log.exception("voice session %s: saving the user message failed", self.id)
            await self._error("storage", f"could not save the message: {_describe(e)}")
            await self._ignored_utterance(done_processing=True)
            return
        self._turn_ids += 1
        agent = AgentTurn(self._turn_ids, turn)
        agent.stop.user = user
        agent.marks.update(
            speech_end=ended.ended_at - utterance.silence_ms / 1000,
            end_of_turn=ended.ended_at,
            transcript=ended.ended_at + stt_ms / 1000,
        )
        # "stop" came after this utterance ended (while it was transcribed, or its message saved): it is saved, not
        # answered (§3.10). Checked after the last await before the turn becomes active, so no stop can slip through.
        if self._stop_at is not None and ended.ended_at <= self._stop_at:
            agent.interrupted = agent.cut = True
            message: Message | None = None
            try:
                message = await self.turns.save_unanswered(turn, heard_text="", reason="stop")
            except Exception as e:
                log.exception("voice session %s: saving the stopped turn failed", self.id)
                await self._error("storage", f"could not save the stopped answer: {_describe(e)}")
            await self._announce(agent, user)
            if message is not None:
                await self._send({"type": "agent_message", "message": _message_json(message)})
            self._processing = False
            await self._set_state(self._current_state())
            return
        self._turn = agent
        await self._set_state("thinking")
        agent.task = self._spawn(self._run_turn(agent), f"{self.id}-turn-{agent.id}")
        self._refresh_hints()  # a document that became READY since: its names for the next utterance

    # -------------------------------------------------------------- the answer

    async def _run_turn(self, agent: AgentTurn) -> None:
        try:
            await self._answer(agent)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # a bug or an unexpected failure: report it and free the session for the next turn
            log.exception("voice session %s: turn %d failed", self.id, agent.id)
            await self._send_for(
                agent, {"type": "error", "detail": f"the answer failed: {_describe(e)}", "stage": "llm"}
            )
            agent.audio_done = True
            await self._finish_turn(agent)

    async def _answer(self, agent: AgentTurn) -> None:
        chunker = SpeechChunker(max_sentences=self.settings.voice.max_spoken_sentences)
        queue: asyncio.Queue[SpeakItem] = asyncio.Queue()
        agent.queue = queue
        speaker = asyncio.create_task(self._speak(agent, queue), name=f"{self.id}-tts-{agent.id}")

        def answered() -> None:
            """The answer's text is complete (a live-data continuation may follow, §3.7): speak its last sentence
            now, not after the continuation."""
            if not agent.errored:
                for text in chunker.flush():
                    queue.put_nowait(text)
            agent.answer_queued = True

        agent.stop.on_answered = answered

        async def on_visual(event: VisualEvent | CanvasEvent) -> None:
            await self._on_visual(agent, event)

        try:
            events = self.turns.run(agent.turn, stop=agent.stop, user=agent.stop.user, on_visual=on_visual)
            async with contextlib.aclosing(events):
                async for event in events:
                    await self._on_turn_event(agent, event, chunker, queue)
            if not agent.errored:  # after an error the cut-off fragment isn't spoken
                for text in chunker.flush():
                    queue.put_nowait(text)
            queue.put_nowait(None)
            await speaker
        finally:
            if not speaker.done():
                speaker.cancel()
                await asyncio.wait([speaker])
        # Every chunk and its frames are out: agent_message closes the turn (the client sends playback_done after it).
        agent.answer_ms = agent.sent_ms
        agent.audio_done = True
        if not agent.visual_released:  # no audio at all (speech failed, or the answer did)
            if agent.errored or agent.cut or agent.message is None:
                self._withdraw_visual(agent)
            else:
                await self._release_visual(agent)
        if (
            agent.tail_wanted
            and not agent.cut
            and agent.first_audio_at is not None
            and not agent.errored
            and not _said_the_chart(agent)
        ):
            await self._send_tail(agent)  # the visual became ready while the answer was spoken (§12.1)
        if agent.message is not None:
            agent.message_sent = True
            await self._send_for(agent, {"type": "agent_message", "message": _message_json(agent.message)})
        self._log_latency(agent)
        if agent.chunks:
            self._arm_playback_timer(agent)
        else:
            await self._finish_turn(agent)

    async def _on_turn_event(
        self, agent: AgentTurn, event: ChatEvent, chunker: SpeechChunker, queue: asyncio.Queue[SpeakItem]
    ) -> None:
        match event:
            case UserMessageEvent(message=message):
                agent.marks["user_message"] = time.perf_counter()
                await self._announce(agent, message)
            case ToolEvent(phase=phase):
                await self._send_for(agent, {"type": "tool", "turn_id": agent.id, **event.payload()})
                if phase == "start" and not agent.chunks:  # the filler, spoken while the web is searched (§3.7)
                    agent.marks["tool_start"] = time.perf_counter()
                    await self._speak_filler(agent)  # sent before the turn's next message (sources)
            case SourcesEvent():
                await self._send_for(agent, {"type": "sources", **event.payload()})
            case DeltaEvent(text=text):
                agent.marks.setdefault("first_delta", time.perf_counter())
                # After the answer is complete (answered()), a delta is a live-data continuation: one whole sentence
                # (§3.7). It is queued for speech before it is sent, and flushed at once rather than at the turn's
                # end, so the answer can't count as "heard" while a continuation is still unspoken.
                continuation = agent.answer_queued
                if continuation:
                    for chunk in (*chunker.feed(text), *chunker.flush()):
                        queue.put_nowait(chunk)
                await self._send_for(agent, {"type": "delta", "turn_id": agent.id, "text": text})
                if not continuation:
                    for chunk in chunker.feed(text):
                        queue.put_nowait(chunk)
            case AgentMessageEvent(message=message):
                agent.message = message
            case VisualEvent() | CanvasEvent():  # a canvas edit's (§12.1): before its "Done."
                await self._send_for(agent, {"type": event.name, "turn_id": agent.id, **event.payload()})
            case ErrorEvent(stage=stage, detail=detail):
                agent.errored = True
                await self._send_for(agent, {"type": "error", "detail": detail, "stage": stage})

    # -------------------------------------------------------------- the canvas (§12.1)

    def _cancel_visual(self) -> None:
        visual = visual_in_progress(self.chat.id)
        if visual is not None:
            visual.cancel()

    async def _on_visual(self, agent: AgentTurn, event: VisualEvent | CanvasEvent) -> None:
        """The answer's visual, as it comes (``visual`` / ``canvas`` with the turn's id; possibly after its
        agent_message, and even after the turn was cut: a visual being planned is cancelled by the next turn that
        needs the model, not by the cut). Its draft is ready while the answer is still being written (§12.1): it is
        held until the answer's first audio, then sent at once (``_release_visual``; §3.10: never before the first
        audio). When it is ready while the answer is still being heard, say so."""
        if not agent.visual_released:
            if not agent.cut:
                agent.held_visual.append(event)
            return
        await self._send_visual(agent, event)

    async def _release_visual(self, agent: AgentTurn) -> None:
        """The answer's first audio is out (or its audio is done without any): its visual's held events now, in order,
        then the rest as they come. A draft already withdrawn by then (the answer turned out to abstain before its
        first audio) is never shown: only a requested one's failure note is left to send."""
        visual = agent.stop.visual
        if visual is not None and visual.status != "ready" and agent.held_visual:
            agent.held_visual = [
                e
                for e in agent.held_visual
                if isinstance(e, VisualEvent) and e.phase != "ready" and visual.status != "cancelled"
            ]
        while agent.held_visual and not agent.cut:
            await self._send_visual(agent, agent.held_visual.pop(0))
        agent.visual_released = not agent.cut

    def _withdraw_visual(self, agent: AgentTurn) -> None:
        """A turn cut (or failed) before its first audio: the client never saw its visual, so there is none."""
        agent.held_visual.clear()
        if agent.stop.visual is not None and not agent.visual_released:
            agent.stop.visual.withdraw()

    async def _send_visual(self, agent: AgentTurn, event: VisualEvent | CanvasEvent) -> None:
        await self._send({"type": event.name, "turn_id": agent.id, **event.payload()})
        if isinstance(event, VisualEvent) and event.phase == "ready" and agent.stop.visual is not None:
            agent.stop.visual.delivered = True
        language = agent.voice_language or agent.turn.language
        if language not in self._tails:  # synthesized while the visual is planned (seconds), once per language
            self._tails.add(language)
            self._spawn(self._warm_tail(language), f"{self.id}-tail-audio")
        if isinstance(event, VisualEvent) and event.phase == "ready":
            visual = agent.stop.visual
            if visual is not None and visual.status != "ready":
                return  # withdrawn meanwhile: nothing to announce
            if visual is not None and visual.trace.draft == "refine" and not visual.done:
                # A draft the planner is still to check (it may withdraw it): the tail waits for its verdict, so it
                # never announces a visual that goes. (Until the visual is done, not only until the planner has
                # answered: its "none" is recorded just before the withdrawal reaches the visual's status, and an
                # answer held while it is checked (services/answer_guard.py) can make its first audio land between.)
                visual.when_settled(lambda v: self._tail_once_settled(agent, v))
            else:
                await self._visual_tail(agent)

    async def _tail_once_settled(self, agent: AgentTurn, visual: TurnVisual) -> None:
        if visual.status == "ready":
            await self._visual_tail(agent)

    async def _warm_tail(self, language: Language) -> None:
        with contextlib.suppress(Exception):  # it is synthesized when needed instead
            await filler_audio(self.tts).phrase(VISUAL_TAILS[language], language)

    async def _visual_tail(self, agent: AgentTurn) -> None:
        """ "It's on screen now." / "स्क्रीन पर दिखा दिया है।" after the answer, only while its audio is still being
        sent or played, the user isn't talking and the turn wasn't cut; otherwise nothing (the chart appearing says
        it). Never before the visual is ready, so it never promises one that fails."""
        if agent.tail_sent or agent.cut or agent.interrupted or agent.errored or agent.tts_failed:
            return
        if self._turn is not agent:  # its audio is over: the next turn (or nothing) is going on
            return
        if self._pending is not None or self._endpointer.in_utterance:  # the user is talking
            return
        if not agent.audio_done:
            agent.tail_wanted = True  # after the answer's last chunk, before agent_message
            return
        if (
            agent.first_audio_at is None
            or time.perf_counter() >= agent.play_end - TAIL_MARGIN_S
            or _said_the_chart(agent)
        ):
            return
        await self._send_tail(agent)
        if agent.playback_timer is not None:
            self._arm_playback_timer(agent)

    async def _send_tail(self, agent: AgentTurn) -> None:
        agent.tail_sent = True
        language = agent.voice_language or agent.turn.language
        text = VISUAL_TAILS[language]
        try:
            pcm = await filler_audio(self.tts).phrase(text, language)
        except Exception as e:
            log.warning("voice session %s: the visual's tail couldn't be synthesized: %s", self.id, _describe(e))
            return
        await self._send_chunk(agent, text, pcm, tail=True)

    async def _announce(self, agent: AgentTurn, user: Message) -> None:
        """``user_message`` then ``turn``, once each (also called for a turn cut before they went out)."""
        async with self._send_lock:
            if not agent.user_sent:
                agent.user_sent = True
                await self._send_text_unlocked(dumps({"type": "user_message", "message": _message_json(user)}))
            if not agent.turn_sent:
                agent.turn_sent = True
                await self._send_text_unlocked(dumps({"type": "turn", "turn_id": agent.id}))

    async def _speak(self, agent: AgentTurn, queue: asyncio.Queue[SpeakItem]) -> None:
        """Synthesize chunks in order and send each: ``audio_chunk``, then its frames, one send-lock acquisition per
        frame (control messages and speech events aren't held up behind a whole chunk)."""
        while (text := await queue.get()) is not None:
            for piece in synthesis_pieces(text):  # a long sentence in clauses: its first words are heard sooner
                if agent.tts_failed or agent.cut:
                    break  # the text is still shown; one error per turn is enough
                agent.tts_busy = True
                try:
                    try:
                        pcm = await self.tts.synthesize(piece, self._voice_for(agent, piece))
                    except Exception as e:
                        agent.tts_failed = True
                        log.warning("voice session %s: speech synthesis failed: %s", self.id, _describe(e))
                        detail = f"speech synthesis failed: {_describe(e)}"
                        await self._send_for(agent, {"type": "error", "detail": detail, "stage": "tts"})
                        break
                    await self._send_chunk(agent, piece, pcm)
                finally:
                    agent.tts_busy = False

    @staticmethod
    def _voice_for(agent: AgentTurn, text: str) -> Language:
        """The voice for a chunk of the answer: the script of the answer's first chunk that has letters (B5: an
        answer asked for in English but written in Hindi is spoken by the Hindi voice), kept for the whole answer, so
        "EBITDA 21.0%" in a Hindi answer doesn't switch voices."""
        if agent.voice_language is None:
            agent.voice_language = script_language(text)
        return agent.voice_language or agent.turn.language

    async def _speak_filler(self, agent: AgentTurn) -> None:
        """The filler of a web search (§3.7), pre-synthesized: sent at once, before the turn's next message. If its
        audio can't be had it is skipped (the answer is still spoken)."""
        language = agent.turn.language
        fillers = filler_audio(self.tts)
        agent.tts_busy = True
        try:
            pcm = await fillers.get(language)
        except Exception as e:
            log.warning("voice session %s: the filler couldn't be synthesized: %s", self.id, _describe(e))
            return
        else:
            await self._send_chunk(agent, fillers.text(language), pcm, filler=True)
        finally:
            agent.tts_busy = False

    async def _send_chunk(
        self, agent: AgentTurn, text: str, pcm: bytes, *, filler: bool = False, tail: bool = False
    ) -> None:
        """``audio_chunk`` (``filler: true`` for the filler, ``tail: true`` for the visual's tail), then its frames;
        ``state: speaking`` with the answer's first chunk (the filler alone doesn't count). Neither the filler nor
        the tail is part of the answer: their words never count in ``heard_text``."""
        if not pcm:
            return
        rate = self.tts.sample_rate
        duration_ms = len(pcm) / 2 / rate * 1000
        async with self._send_lock:
            if agent.cut:  # synthesized after the decision: dropped
                return
            index = len(agent.chunks)
            first_answer_chunk = not filler and not tail and agent.first_audio_at is None
            chunk = SpokenChunk(index, text, agent.sent_ms, duration_ms, filler=filler or tail)
            announce: dict[str, Any] = {
                "type": "audio_chunk",
                "turn_id": agent.id,
                "chunk_index": index,
                "text": text,
                "duration_ms": round(duration_ms, 1),
            }
            if filler:
                announce["filler"] = True
            if tail:
                announce["tail"] = True
            await self._send_text_unlocked(dumps(announce))
            agent.chunks.append(chunk)
            agent.sent_ms = chunk.end_ms
            # The client plays a chunk when it arrives or when the previous one ends, whichever is later.
            now = time.perf_counter()
            start = max(agent.play_end, now)
            agent.play_starts.append(start)
            agent.play_end = start + duration_ms / 1000
            if filler:
                agent.marks["filler_audio"], agent.marks["filler_end"] = now, agent.play_end
            elif first_answer_chunk:
                agent.first_audio_at = agent.marks["first_audio"] = now
        for frame in audio_frames(agent.id, index, pcm, sample_rate=rate):
            async with self._send_lock:
                if agent.cut:
                    break
                await self._send_bytes_unlocked(frame)
        if first_answer_chunk and not agent.cut:
            await self._set_state("speaking")
            await self._release_visual(agent)  # the draft, ready since the sources: on screen as the answer starts

    def _answer_heard(self, agent: AgentTurn, played_ms: float | None) -> bool:
        """The answer is complete, all of it was spoken, and the client has played it: only a live-data continuation
        was still to come (§3.7)."""
        if not (agent.stop.answered and agent.answer_queued) or agent.tts_busy:
            return False
        if agent.queue is not None and not agent.queue.empty():
            return False
        return self._played_ms(agent, played_ms) >= _answer_end_ms(agent) - 1

    def _arm_playback_timer(self, agent: AgentTurn) -> None:
        """End the turn when the client says playback_done, or PLAYBACK_GRACE_S after its audio should have finished
        playing: the expected end follows the chunks as they were sent (gaps included) and the client's progress."""
        if agent.playback_timer is not None:
            agent.playback_timer.cancel()
        delay = max(0.0, agent.play_end + PLAYBACK_GRACE_S - time.perf_counter())
        agent.playback_timer = asyncio.get_running_loop().call_later(delay, self._playback_timer_fired, agent)

    def _playback_timer_fired(self, agent: AgentTurn) -> None:
        agent.playback_timer = None
        if self._turn is not agent:
            return
        if time.perf_counter() < agent.play_end + PLAYBACK_GRACE_S - 0.01:  # a progress report moved the end
            self._arm_playback_timer(agent)
            return
        self._spawn(self._finish_turn(agent), f"{self.id}-playback-end")

    async def _finish_turn(self, agent: AgentTurn) -> None:
        if agent.playback_timer is not None:
            agent.playback_timer.cancel()
            agent.playback_timer = None
        if self._turn is not agent:
            return
        self._turn = None
        if self._pending is not None and self._pending.agent is agent:
            await self._decide(self._pending, "resume")  # the answer finished while deciding: nothing to stop
        await self._set_state(self._current_state())

    def _played_ms(self, agent: AgentTurn, explicit: float | None) -> float:
        """How much of the turn's audio the user heard: the client's number when it gave one, else its last progress
        report plus the time since, else what the client should have played by now (each chunk from when it arrived
        or the previous one ended); never more than was sent."""
        if explicit is not None:
            return min(explicit, agent.sent_ms)
        now = time.perf_counter()
        if agent.reported_ms is not None:
            return min(agent.sent_ms, agent.reported_ms + (now - agent.reported_at) * 1000)
        played = sum(
            min(max(0.0, (now - start) * 1000), chunk.duration_ms)
            for start, chunk in zip(agent.play_starts, agent.chunks, strict=True)
        )
        return min(agent.sent_ms, played)

    # -------------------------------------------------------------- interruptions

    def _cut(self, agent: AgentTurn) -> bool:
        """Mute a turn at once (synchronously, before anything is announced): from now on none of its sources,
        deltas, audio chunks or frames are sent, and it is no longer the active turn. True if a barge-in decision for
        it was pending (the client is waiting for one)."""
        agent.cut = True
        self._withdraw_visual(agent)  # (only if its first audio never went out)
        if agent.playback_timer is not None:
            agent.playback_timer.cancel()
            agent.playback_timer = None
        if self._turn is agent:
            self._turn = None
        pending = self._pending
        if pending is None or pending.agent is not agent:
            return False
        self._pending = None
        if pending.deadline is not None and pending.deadline is not asyncio.current_task():
            pending.deadline.cancel()
        return True

    async def _interrupt(
        self,
        agent: AgentTurn,
        reason: InterruptReason,
        played_ms: float | None,
        *,
        decision: bool = False,
        finished: bool = False,
    ) -> None:
        """Cut an answer short: mute it, cancel generation and speech, then settle it (the decision if one is owed,
        the answer saved with what was heard, ``agent_message``) in its own task that the next turn waits for.

        ``finished``: the answer was complete and heard, only its live-data continuation was still to come; the turn
        just ends (no "interrupted" state, the answer saved complete).

        Everything up to starting that task is synchronous, so no other task can slip in between the cut and the
        settle: neither another frame of this turn nor the next turn's user message."""
        if agent.interrupted:
            return
        agent.interrupted = True
        heard_all = finished or self._answer_heard(agent, played_ms)
        decision = (
            self._cut(agent) or decision
        )  # e.g. "stop" while a barge-in was being decided: the client is owed one
        played = self._played_ms(agent, played_ms)
        heard = heard_text(agent.chunks, played)
        agent.stop.heard_text, agent.stop.reason = heard, reason
        if agent.task is not None and not agent.task.done():
            agent.task.cancel()  # generation and speech stop now, whatever happens to the rest
        settle = asyncio.ensure_future(self._settle(agent, reason, played, heard, decision, heard_all, finished))
        self._settling.add(settle)
        settle.add_done_callback(self._settling.discard)
        await asyncio.shield(settle)

    async def _settle(
        self,
        agent: AgentTurn,
        reason: InterruptReason,
        played: float,
        heard: str,
        decision: bool,
        heard_all: bool = False,
        finished: bool = False,
    ) -> None:
        async with self._turn_lock:
            if decision:
                await self._send({"type": "barge_in", "turn_id": agent.id, "decision": "stop"})
            if not finished:
                await self._set_state("interrupted")
            if agent.task is not None:
                await asyncio.wait([agent.task])  # the pipeline saves the stopped answer before the task ends
            message = agent.stop.saved
            complete = agent.message or agent.stop.completed
            # Fully heard: all of it was played (the visual's tail after it doesn't count), and either the turn had
            # ended or only a continuation was to come.
            fully_heard = played >= _answer_end_ms(agent) and (agent.audio_done or heard_all)
            try:
                if message is None and complete is not None:
                    if not fully_heard:  # fully heard: it was sent complete already and nothing was cut
                        message = await self.messages.record_interruption(complete.id, heard_text=heard, reason=reason)
                    elif not agent.message_sent:  # complete and heard, but agent_message hadn't gone out yet
                        message = complete
                elif message is None and agent.stop.user is not None:  # stopped before the answer had any text
                    message = await self.turns.save_unanswered(agent.turn, heard_text=heard, reason=reason)
            except Exception as e:
                log.exception("voice session %s: saving the interrupted answer failed", self.id)
                await self._error("storage", f"could not save the interrupted answer: {_describe(e)}")
            if message is not None:
                if agent.stop.user is not None:
                    await self._announce(agent, agent.stop.user)  # if the cut came before they went out
                await self._send({"type": "agent_message", "message": _message_json(message)})
            log.info(
                "voice session %s: turn %d cut (%s) after %.0f ms of audio (heard %d words)",
                self.id,
                agent.id,
                reason,
                played,
                len(heard.split()),
            )
        await self._set_state(self._current_state())

    # -------------------------------------------------------------- barge-in

    async def _on_barge_in_start(self, message: BargeInStart) -> None:
        agent = self._turn
        if agent is None or agent.id != message.turn_id or agent.interrupted or self._input_mode == "ptt":
            await self._send({"type": "barge_in", "turn_id": message.turn_id, "decision": "resume"})
            return
        if self._pending is not None:
            return  # already deciding
        agent.barge_in_ms, agent.barge_in_at = message.played_ms, time.perf_counter()
        pending = PendingBargeIn(agent, message.played_ms)
        self._pending = pending
        pending.deadline = self._spawn(self._barge_in_deadline(pending), f"{self.id}-barge-in-deadline")
        await self._barge_in_check()

    async def _barge_in_deadline(self, pending: PendingBargeIn) -> None:
        timeout = self.barge_in.decision_timeout_ms / 1000
        await asyncio.sleep(timeout)
        pending.deadline_passed = True
        await self._evaluate(pending)
        if self._pending is not pending:
            return
        # Still talking, but only acknowledgements so far, or the transcription isn't in yet: listen longer, with a
        # transcript of everything said (when one is running, its result asks for the next).
        if not pending.transcribing:
            self._transcribe_for(pending)
        await asyncio.sleep(timeout * (ACKNOWLEDGEMENT_GRACE - 1))
        pending.cap_passed = True
        await self._evaluate(pending)
        if self._pending is pending and pending.ended_speech_ms is None:  # decided by the cap, or by the worker
            await self._decide(pending, "resume")

    async def _barge_in_check(self) -> None:
        """After new audio: transcribe the interrupting speech once it is long enough, and re-evaluate."""
        pending = self._pending
        if pending is None:
            await self._watch_speech()
            return
        ep = self._endpointer
        if pending.stt is None and ep.in_utterance and ep.speech_ms >= ep.min_speech:
            pending.stt = self._transcribe_for(pending)
        await self._evaluate(pending)

    def _transcribe_for(self, pending: PendingBargeIn) -> asyncio.Task[None] | None:
        """Transcribe the interrupting speech so far (again, while it goes on past the deadline). Counted as running
        until its result is in, so the deadline waits for it (B3)."""
        if not self._endpointer.in_utterance:
            return None
        pending.transcribing += 1
        audio = self._endpointer.snapshot()
        return self._spawn(self._barge_in_transcript(pending, audio), f"{self.id}-barge-in-stt")

    async def _barge_in_transcript(self, pending: PendingBargeIn, audio: np.ndarray) -> None:
        try:
            transcript = await self._transcribe(audio, report=False, hinted=False)
        finally:
            pending.transcribing -= 1
        if self._pending is not pending:
            return
        if transcript is not None:
            if transcript.text and not self._background(transcript):
                await self._send({"type": "transcript_partial", "text": transcript.text})
            self._note_transcript(pending, transcript)
        await self._evaluate(pending)
        ep = self._endpointer
        if self._pending is pending and pending.deadline_passed and ep.speaking and not pending.transcribing:
            self._transcribe_for(pending)  # still talking past the deadline: a fuller transcript

    def _note_transcript(self, pending: PendingBargeIn, transcript: Transcript) -> None:
        text = transcript.text
        pending.transcript = text
        if self._background(transcript):  # the TV over the answer: noise, however many words
            pending.backchannel, pending.real_words, pending.stop_words = True, 0, False
            return
        pending.backchannel = is_backchannel(text, self.barge_in.backchannel_max_words)
        pending.real_words = real_words(text)
        pending.stop_words = is_stop_phrase(text)

    async def _evaluate(self, pending: PendingBargeIn) -> None:
        if self._pending is not pending:
            return
        if pending.ended_speech_ms is not None:
            # The utterance is over and its final transcript is coming from the worker, which decides if nothing has:
            # a partial transcript decides only when it is sure. A short "Stop." said and over before any transcript
            # was in used to get "resume" here (its speech reads 0 once the utterance is closed), then "stop".
            sure_stop = pending.backchannel is False and pending.real_words >= 2
            if pending.transcript is not None and (pending.stop_words or sure_stop):
                await self._decide(pending, "stop")
            elif pending.transcript is not None and pending.backchannel:
                await self._decide(pending, "resume")
            return
        ep = self._endpointer
        evidence = BargeInEvidence(
            speech_ms=ep.speech_ms,
            speaking=ep.speaking,
            transcript=pending.transcript,
            ended=False,
            deadline_passed=pending.deadline_passed,
            real_words=pending.real_words,
            cap_passed=pending.cap_passed,
            transcribing=pending.transcribing > 0,
            stop_words=pending.stop_words,
        )
        verdict = barge_in_verdict(evidence, min_speech_ms=ep.min_speech, is_backchannel=pending.backchannel)
        if verdict is not None:
            await self._decide(pending, verdict)

    async def _decide(self, pending: PendingBargeIn, verdict: Verdict) -> None:
        if self._pending is not pending:
            return
        self._pending = None
        if pending.deadline is not None and pending.deadline is not asyncio.current_task():
            pending.deadline.cancel()
        if verdict == "stop":  # the decision is sent by the settle, after the turn is muted
            await self._interrupt(pending.agent, "barge_in", pending.played_ms, decision=True)
        else:
            # The answer goes on at full volume: if this speech turns out to be an interruption after all (its final
            # transcript has real words), what was heard is whatever has been played by then (the playback reports),
            # not what had been played when the speech began. Speech that goes on is watched from here (B3).
            pending.agent.barge_in_ms = pending.agent.barge_in_at = None
            ep = self._endpointer
            if ep.in_utterance:
                self._watch = SpeechWatch(pending.agent, self._utterance_no, ep.speech_ms)
            await self._send({"type": "barge_in", "turn_id": pending.agent.id, "decision": "resume"})

    # -------------------------------------------------------------- speech nobody is deciding about (B3)

    async def _watch_speech(self) -> None:
        """Speech while the agent answers and no barge-in decision is pending: transcribe it every WATCH_STEP_MS of
        new speech (from decision_timeout_ms of speech, or right after a "resume"), so real words stop the answer
        while the user is still talking."""
        agent, ep = self._turn, self._endpointer
        if agent is None or agent.interrupted or agent.cut or not ep.in_utterance or not ep.speaking:
            return
        watch = self._watch
        if watch is None or watch.agent is not agent or watch.utterance != self._utterance_no:
            first = max(self.vad_settings.min_speech_ms, self.barge_in.decision_timeout_ms) - WATCH_STEP_MS
            watch = self._watch = SpeechWatch(agent, self._utterance_no, first)
        if watch.stt is not None and not watch.stt.done():
            return
        if ep.speech_ms < watch.checked_ms + WATCH_STEP_MS:
            return
        watch.checked_ms = ep.speech_ms
        watch.stt = self._spawn(self._watch_transcript(watch, ep.snapshot()), f"{self.id}-watch-stt")

    async def _watch_transcript(self, watch: SpeechWatch, audio: np.ndarray) -> None:
        transcript = await self._transcribe(audio, report=False, hinted=False)
        agent = watch.agent
        if transcript is None or not transcript.text or self._turn is not agent or agent.interrupted:
            return
        await self._send({"type": "transcript_partial", "text": transcript.text})
        pending = self._pending
        if pending is not None:  # the client asked meanwhile: this is evidence for its decision
            if pending.agent is agent:
                self._note_transcript(pending, transcript)
                await self._evaluate(pending)
            return
        text = transcript.text
        if self._background(transcript):  # background speech over the answer: not an interruption (§3.10)
            return
        stop = is_stop_phrase(text)  # "Stop." over the answer: stopped now, not when the utterance ends
        if not stop and (is_backchannel(text, self.barge_in.backchannel_max_words) or real_words(text) < 2):
            return
        if self._answer_heard(agent, None):  # complete and heard: the end of the utterance ends the turn (§3.7)
            return
        log.info("voice session %s: speech over turn %d has real words (%r): stopping it", self.id, agent.id, text)
        await self._interrupt(agent, "barge_in", None, decision=True)

    # -------------------------------------------------------------- observability

    def _log_latency(self, agent: AgentTurn) -> None:
        marks = agent.marks
        start = marks.get("speech_end")
        if start is None:
            return

        def at(name: str) -> str:
            return f"{(marks[name] - start) * 1000:.0f}" if name in marks else "-"

        log.info(
            "voice session %s turn %d latency from end of speech (ms): end-of-turn %s, transcript %s, "
            "user_message %s, first delta %s, first audio %s",
            self.id,
            agent.id,
            at("end_of_turn"),
            at("transcript"),
            at("user_message"),
            at("first_delta"),
            at("first_audio"),
        )
        gap = filler_gap_ms(agent)
        if gap is not None:
            log.info(
                "voice session %s turn %d web search: filler audio at %s ms, its end at %s ms, first answer audio "
                "at %s ms: %.0f ms after the filler",
                self.id,
                agent.id,
                at("filler_audio"),
                at("filler_end"),
                at("first_audio"),
                gap,
            )


def _db(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _said_the_chart(agent: AgentTurn) -> bool:
    """The answer's speech already points at the chart ("The chart shows …"): "It's on screen now." would repeat it."""
    return any(_NAMES_THE_CHART.search(c.text) for c in agent.chunks if not c.filler)


SPLIT_MIN_WORDS = 12  # a chunk this long is synthesized in clauses
SPLIT_MIN_PART = 4  # ... none shorter than this
_CLAUSE_CUT = re.compile(r"[,;:](?=\s)")  # "4,210" and "18.2%" never split


def synthesis_pieces(text: str) -> list[str]:
    """``text`` cut at clause boundaries into pieces of at least ``SPLIT_MIN_PART`` words, when it has
    ``SPLIT_MIN_WORDS`` or more: each piece is synthesized and sent as its own chunk, so the first is heard while the
    rest is synthesized (measured: Kokoro takes ~2.8 s on the CPU for a 20-word sentence, ~0.2 s per 5 words). A cut
    goes to the boundary nearest the middle; each part is cut again if still long."""
    words = text.split()
    if len(words) < SPLIT_MIN_WORDS:
        return [text]
    best: tuple[float, int] | None = None
    for m in _CLAUSE_CUT.finditer(text):
        before = len(text[: m.end()].split())
        if before >= SPLIT_MIN_PART and len(words) - before >= SPLIT_MIN_PART:
            score = abs(before - len(words) / 2)
            if best is None or score < best[0]:
                best = (score, m.end())
    if best is None:
        return [text]
    head, tail = text[: best[1]].strip(), text[best[1] :].strip()
    return [*synthesis_pieces(head), *synthesis_pieces(tail)]


def _answer_end_ms(agent: AgentTurn) -> float:
    """Where the answer's audio ends in the turn's audio: the visual's tail, if any, comes after it (§12.1)."""
    return agent.answer_ms if agent.answer_ms is not None else agent.sent_ms


def filler_gap_ms(agent: AgentTurn) -> float | None:
    """How long after the filler finished playing the answer's first audio arrived (§3.7 target: < 1 s); 0 when it
    arrived before the filler ended (it plays right after it). None without a filler or answer audio."""
    marks = agent.marks
    if "filler_end" not in marks or "first_audio" not in marks:
        return None
    return max(0.0, (marks["first_audio"] - marks["filler_end"]) * 1000)


# ------------------------------------------------------------------ registry


class VoiceSessions:
    """The live voice sessions of this process, at most one per chat: opening a second one closes the first (4409).
    ``canvas``: the app's canvas service, so answers get visuals (§12.1)."""

    def __init__(self, container: Container, *, canvas: CanvasService | None = None) -> None:
        self.container = container
        self.canvas = canvas
        self.vocabulary = speech_vocabulary(container)  # shared: a chat's hints outlive a reconnect
        self._sessions: dict[str, VoiceSession] = {}
        self._lock = asyncio.Lock()

    def get(self, chat_id: str) -> VoiceSession | None:
        return self._sessions.get(chat_id)

    async def open(self, chat_id: str, transport: Transport) -> VoiceSession:
        """A session for the chat (unknown chat → NotFound). A session already open for it is closed first, after it
        has saved its state, so the transcript stays in order."""
        db = self.container["metadata_db"]
        assert isinstance(db, MetadataDB)
        chat = await ChatService(db).get(chat_id)
        async with self._lock:
            previous = self._sessions.get(chat_id)
            session = VoiceSession.from_container(
                chat, transport, self.container, canvas=self.canvas, vocabulary=self.vocabulary
            )
            self._sessions[chat_id] = session
        if previous is not None:
            previous.close(CLOSE_REPLACED, "another voice session was opened for this chat")
            await previous.wait_closed()
        return session

    def release(self, session: VoiceSession) -> None:
        if self._sessions.get(session.chat.id) is session:
            del self._sessions[session.chat.id]

    async def close_all(self) -> None:
        sessions = list(self._sessions.values())
        for session in sessions:
            session.close(CLOSE_GOING_AWAY, "server shutting down")
        await asyncio.gather(*(s.wait_closed() for s in sessions))
