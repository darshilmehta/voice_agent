"""A live voice conversation in one chat: the state machine behind ``WS /ws/chats/{chat_id}/voice`` (docs/DESIGN.md
§3.3, §3.5, §3.9).

    LISTENING ──end of turn (server VAD)──► THINKING ──first audio chunk──► SPEAKING ──playback done──► LISTENING
        ▲                                     │   ▲                            │
        │                                     │   └── the next turn starts ◄───┤ barge-in confirmed / "stop":
        └──────────── INTERRUPTED ◄───────────┴────────────────────────────────┘ answer cut, what was heard saved

- Audio in: PCM16 16 kHz → Silero (this session's state) → ``Endpointer``. At half the end-of-turn silence the
  utterance is transcribed speculatively; at the end of the turn that transcript is reused unless speech resumed.
- A user turn is ``ChatTurnService`` with ``modality="voice"``, ``length="short"``: the same pipeline as text chat.
  Its deltas go to the client and through ``SpeechChunker`` to TTS; each chunk is announced (``audio_chunk``) and
  followed by its binary frames. ``agent_message`` is sent once the last chunk is out.
- Barge-in, duck-then-decide: the client ducks and sends ``barge_in_start``; the server decides within
  ``voice.barge_in.decision_timeout_ms`` from its own VAD and a transcript of the new speech (``barge_in_verdict``).
  On "stop" the answer is cancelled (LLM stream closed, queued TTS dropped) and saved with ``heard_text``; the new
  utterance becomes the next user turn when it ends. An utterance that ends while the agent is answering and isn't a
  backchannel stops the answer too, even without ``barge_in_start``.
- One utterance is processed at a time (STT, then the turn), in order; transcript order is preserved by
  ``_turn_lock`` (an interrupted answer is saved before the next user message).
- Failures are reported as ``error {stage}`` and the session goes on.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ...db.types import new_id
from ...domain.projects import Chat, Message
from ...providers.registry import Container
from ...providers.speech import SpeechRecognizer, SpeechSynthesizer, Transcript, VoiceActivityDetector
from ...providers.storage import MetadataDB
from ...settings import Language, Settings
from ..base import InvalidInput, NotFound
from ..chat_turns import (
    AgentMessageEvent,
    AnswerStop,
    ChatEvent,
    ChatTurnService,
    DeltaEvent,
    ErrorEvent,
    InterruptReason,
    SourcesEvent,
    Turn,
    UserMessageEvent,
    detach,
)
from ..chats import ChatService
from ..messages import MessageService
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
    Playback,
    PlaybackDone,
    ProtocolError,
    Start,
    Stop,
    audio_frames,
    dumps,
    parse_client_message,
)
from .speech_text import SpeechChunker, SpokenChunk, heard_text, is_backchannel, is_filler, normalize_utterance
from .turn_taking import (
    FRAME_SAMPLES,
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

log = logging.getLogger(__name__)

PLAYBACK_GRACE_S = 2.0  # a turn whose client never says playback_done ends this long after its audio should have
CLOSE_WAIT_S = 10.0  # how long a replaced session may take to save its state and close


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


@dataclass(eq=False)
class AgentTurn:
    """One answer: generation, speech and playback of the reply to one user utterance."""

    id: int
    turn: Turn
    stop: AnswerStop = field(default_factory=AnswerStop)
    task: asyncio.Task[None] | None = None
    chunks: list[SpokenChunk] = field(default_factory=list)  # announced to the client, in order
    sent_ms: float = 0.0  # audio sent so far
    message: Message | None = None  # the saved, complete answer
    audio_done: bool = False  # generation finished and every chunk was sent
    cut: bool = False  # stopped or barged in: nothing more of this turn (sources, deltas, audio) is sent
    interrupted: bool = False  # the interruption has been handled (or is being handled)
    user_sent: bool = False  # user_message sent to the client
    turn_sent: bool = False  # turn sent to the client
    tts_failed: bool = False
    first_audio_at: float | None = None
    reported_ms: float | None = None  # last playback progress from the client
    reported_at: float = 0.0
    barge_in_ms: float | None = None  # played_ms when the user started talking over the answer
    playback_timer: asyncio.TimerHandle | None = None
    marks: dict[str, float] = field(default_factory=dict)  # perf_counter timestamps for the latency log


@dataclass(eq=False)
class PendingBargeIn:
    agent: AgentTurn
    played_ms: float
    deadline: asyncio.Task[None] | None = None
    stt: asyncio.Task[None] | None = None
    transcript: str | None = None
    backchannel: bool | None = None


@dataclass(eq=False)
class _EndedUtterance:
    utterance: Utterance
    speculative: asyncio.Task[Transcript | None] | None
    ended_at: float  # perf_counter at the end of turn


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
        self._endpointer = Endpointer(
            threshold=settings.vad.threshold,
            min_speech_ms=settings.vad.min_speech_ms,
            end_of_turn_ms=settings.vad.end_of_turn_ms,
        )
        self._language: Language | None = None
        self._started = False
        self._rest = np.zeros(0, dtype=np.float32)
        self._turn: AgentTurn | None = None
        self._turn_ids = 0
        self._turn_lock = asyncio.Lock()
        self._pending: PendingBargeIn | None = None
        self._speculative: asyncio.Task[Transcript | None] | None = None
        self._utterances: asyncio.Queue[_EndedUtterance] = asyncio.Queue()
        self._send_lock = asyncio.Lock()
        self._open = True  # the transport can still be written to
        self._close_requested = asyncio.Event()
        self._close_code = CLOSE_NORMAL
        self._close_reason = ""
        self._finished = asyncio.Event()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._state: AgentState | None = None
        self._warned_early_audio = False

    @classmethod
    def from_container(cls, chat: Chat, transport: Transport, container: Container) -> VoiceSession:
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
            turns=ChatTurnService.from_container(container),
            messages=MessageService(db),
            vad=vad,
            stt=stt,
            tts=tts,
            settings=container.settings,
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
            if self._turn is not None:
                await self._interrupt(self._turn, "stop", None)
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

    async def _error(self, stage: ErrorStage, detail: str) -> None:
        await self._send({"type": "error", "detail": detail, "stage": stage})

    async def _set_state(self, state: AgentState) -> None:
        if state != self._state:
            self._state = state
            await self._send({"type": "state", "state": state})

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
                await self._error("audio", str(e))
                continue
            if isinstance(message, End):
                return
            await self._on_control(message)

    async def _on_control(self, message: Start | BargeInStart | Playback | PlaybackDone | Stop) -> None:
        match message:
            case Start(language=language):
                self._language, self._started = language, True
                await self._send(
                    {
                        "type": "ready",
                        "session_id": self.id,
                        "input_sample_rate": INPUT_SAMPLE_RATE,
                        "output_sample_rate": self.tts.sample_rate,
                        "language": language,
                    }
                )
                self._state = None  # always (re)announce the state after ready
                await self._set_state(self._current_state())
            case BargeInStart():
                await self._on_barge_in_start(message)
            case Playback(turn_id=turn_id, played_ms=played_ms):
                agent = self._turn
                if agent is not None and agent.id == turn_id:
                    agent.reported_ms, agent.reported_at = played_ms, time.perf_counter()
            case PlaybackDone(turn_id=turn_id):
                agent = self._turn
                if agent is not None and agent.id == turn_id and agent.audio_done:
                    await self._finish_turn(agent)
            case Stop():
                if self._turn is not None:
                    await self._interrupt(self._turn, "stop", None)

    def _current_state(self) -> AgentState:
        agent = self._turn
        if agent is None:
            return "listening"
        return "speaking" if agent.chunks else "thinking"

    def _languages(self) -> list[Language]:
        return [self._language] if self._language else list(self.settings.stt.languages)

    # -------------------------------------------------------------- audio in

    async def _on_audio(self, data: bytes) -> None:
        if not self._started:
            if not self._warned_early_audio:
                self._warned_early_audio = True
                await self._error("audio", 'audio before start is ignored: send {"type": "start"} first')
            return
        if len(data) % 2 or len(data) > MAX_INPUT_FRAME_BYTES:
            await self._error("audio", f"audio frames must be PCM16 mono 16 kHz, at most 1 s (got {len(data)} bytes)")
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
            await self._error("audio", f"voice activity detection failed: {_describe(e)}")
            return
        for frame, probability in zip(frames, probabilities, strict=True):
            for event in self._endpointer.push(frame, probability):
                await self._on_vad_event(event)
        await self._barge_in_check()

    async def _on_vad_event(self, event: VADEvent) -> None:
        match event:
            case SpeechStarted():
                await self._send({"type": "user_speech", "phase": "start"})
            case SpeechPaused(audio=audio):
                self._drop_speculative()
                self._speculative = self._spawn(self._transcribe(audio, partial=True), f"{self.id}-speculative-stt")
            case SpeechResumed():
                self._drop_speculative()
            case SpeechEnded(utterance=utterance):
                speculative, self._speculative = self._speculative, None
                await self._send({"type": "user_speech", "phase": "end"})
                if self._turn is None:
                    await self._set_state("thinking")
                self._utterances.put_nowait(_EndedUtterance(utterance, speculative, time.perf_counter()))
            case SpeechDiscarded():
                self._drop_speculative()
                await self._send({"type": "user_speech", "phase": "end"})
                if self._pending is not None:
                    await self._decide(self._pending, "resume")

    def _drop_speculative(self) -> None:
        # A running transcription can't be interrupted (it runs in a thread); its result is simply not used.
        self._speculative = None

    async def _transcribe(self, audio: np.ndarray, *, partial: bool) -> Transcript | None:
        try:
            transcript = await self.stt.transcribe(audio, self._languages())
        except Exception as e:
            log.warning("voice session %s: transcription failed: %s", self.id, _describe(e))
            if not partial:
                await self._error("stt", f"transcription failed: {_describe(e)}")
            return None
        if partial and transcript.text:
            await self._send({"type": "transcript_partial", "text": transcript.text})
        return transcript

    # -------------------------------------------------------------- utterances → turns

    async def _utterance_worker(self) -> None:
        while True:
            ended = await self._utterances.get()
            await self._handle_utterance(ended)

    async def _handle_utterance(self, ended: _EndedUtterance) -> None:
        transcript: Transcript | None = None
        speculative = False
        if ended.speculative is not None:
            await asyncio.wait([ended.speculative])  # never raises; this task's own cancellation still propagates
            if not ended.speculative.cancelled():
                transcript = ended.speculative.result()
            speculative = transcript is not None
        if transcript is None:
            transcript = await self._transcribe(ended.utterance.audio, partial=False)
        stt_ms = (time.perf_counter() - ended.ended_at) * 1000
        agent = self._turn
        if transcript is None:  # STT failed (reported)
            if self._pending is not None:
                await self._decide(self._pending, "resume")
            if agent is None:
                await self._set_state("listening")
            return

        text = transcript.text.strip()
        if agent is not None and not agent.interrupted:
            # The agent is answering: is this an interruption?
            if is_backchannel(text, self.barge_in.backchannel_max_words):
                if self._pending is not None:
                    await self._decide(self._pending, "resume")
                return  # "mm-hmm", "okay": the agent goes on
            if self._pending is not None and self._pending.agent is agent:
                await self._decide(self._pending, "stop")
            else:
                self._cut(agent)  # nothing more of it is sent once the decision is out
                await self._send({"type": "barge_in", "turn_id": agent.id, "decision": "stop"})
                await self._interrupt(agent, "barge_in", agent.barge_in_ms)
        if not normalize_utterance(text) or is_filler(text):  # noise, or only "hmm"/"mm-hmm": nothing to answer
            if self._turn is None:
                await self._set_state("listening")
            return
        await self._start_turn(transcript, ended, stt_ms, speculative)

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
        async with self._turn_lock:  # an interrupted answer is saved before this user message
            try:
                turn = await self.turns.begin(
                    self.chat.id,
                    transcript.text,
                    language=transcript.language,
                    modality="voice",
                    length="short",
                    input_language=transcript.language,
                    input_latency=latency,
                )
            except NotFound:
                await self._error("storage", "this chat no longer exists")
                self.close(CLOSE_CHAT_NOT_FOUND, "chat not found")
                return
            except InvalidInput as e:
                await self._error("stt", f"transcript not usable: {e}")
                await self._set_state("listening")
                return
            self._turn_ids += 1
            agent = AgentTurn(self._turn_ids, turn)
            agent.marks.update(
                speech_end=ended.ended_at - utterance.silence_ms / 1000,
                end_of_turn=ended.ended_at,
                transcript=ended.ended_at + stt_ms / 1000,
            )
            self._turn = agent
            await self._set_state("thinking")
            agent.task = asyncio.create_task(self._run_turn(agent), name=f"{self.id}-turn-{agent.id}")

    # -------------------------------------------------------------- the answer

    async def _run_turn(self, agent: AgentTurn) -> None:
        chunker = SpeechChunker(max_sentences=self.settings.voice.max_spoken_sentences)
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        speaker = asyncio.create_task(self._speak(agent, queue), name=f"{self.id}-tts-{agent.id}")
        try:
            async with contextlib.aclosing(self.turns.run(agent.turn, stop=agent.stop)) as events:
                async for event in events:
                    await self._on_turn_event(agent, event, chunker, queue)
            for text in chunker.flush():
                queue.put_nowait(text)
            queue.put_nowait(None)
            await speaker
        finally:
            if not speaker.done():
                speaker.cancel()
                await asyncio.wait([speaker])
        # Every chunk and its frames are out: agent_message closes the turn (the client sends playback_done after it).
        agent.audio_done = True
        if agent.message is not None:
            await self._send_for(agent, {"type": "agent_message", "message": _message_json(agent.message)})
        self._log_latency(agent)
        if agent.chunks:
            self._schedule_playback_end(agent)
        else:
            await self._finish_turn(agent)

    async def _on_turn_event(
        self, agent: AgentTurn, event: ChatEvent, chunker: SpeechChunker, queue: asyncio.Queue[str | None]
    ) -> None:
        match event:
            case UserMessageEvent(message=message):
                agent.marks["user_message"] = time.perf_counter()
                await self._announce(agent, message)
            case SourcesEvent():
                await self._send_for(agent, {"type": "sources", **event.payload()})
            case DeltaEvent(text=text):
                agent.marks.setdefault("first_delta", time.perf_counter())
                await self._send_for(agent, {"type": "delta", "turn_id": agent.id, "text": text})
                for chunk in chunker.feed(text):
                    queue.put_nowait(chunk)
            case AgentMessageEvent(message=message):
                agent.message = message
            case ErrorEvent(stage=stage, detail=detail):
                await self._send_for(agent, {"type": "error", "detail": detail, "stage": stage})

    async def _announce(self, agent: AgentTurn, user: Message) -> None:
        """``user_message`` then ``turn``, once each (also called for a turn cut before they went out)."""
        async with self._send_lock:
            if not agent.user_sent:
                agent.user_sent = True
                await self._send_text_unlocked(dumps({"type": "user_message", "message": _message_json(user)}))
            if not agent.turn_sent:
                agent.turn_sent = True
                await self._send_text_unlocked(dumps({"type": "turn", "turn_id": agent.id}))

    async def _send_for(self, agent: AgentTurn, message: dict[str, Any]) -> None:
        """Send a message of a turn unless the turn has been cut (checked under the send lock, so nothing of a cut
        turn follows the barge_in decision or the reaction to stop)."""
        async with self._send_lock:
            if not agent.cut:
                await self._send_text_unlocked(dumps(message))

    async def _speak(self, agent: AgentTurn, queue: asyncio.Queue[str | None]) -> None:
        """Synthesize chunks in order and send each: ``audio_chunk`` then its frames, as one unit."""
        rate = self.tts.sample_rate
        while (text := await queue.get()) is not None:
            if agent.tts_failed or agent.cut:
                continue  # the text is still shown; one error per turn is enough
            try:
                pcm = await self.tts.synthesize(text, agent.turn.language)
            except Exception as e:
                agent.tts_failed = True
                log.warning("voice session %s: speech synthesis failed: %s", self.id, _describe(e))
                await self._send_for(
                    agent, {"type": "error", "detail": f"speech synthesis failed: {_describe(e)}", "stage": "tts"}
                )
                continue
            if not pcm:
                continue
            index = len(agent.chunks)
            chunk = SpokenChunk(index, text, agent.sent_ms, len(pcm) / 2 / rate * 1000)
            frames = audio_frames(agent.id, index, pcm, sample_rate=rate)
            announce = {
                "type": "audio_chunk",
                "turn_id": agent.id,
                "chunk_index": index,
                "text": text,
                "duration_ms": round(chunk.duration_ms, 1),
            }
            async with self._send_lock:
                if agent.cut:  # synthesized after the decision: dropped
                    continue
                await self._send_text_unlocked(dumps(announce))
                agent.chunks.append(chunk)
                agent.sent_ms = chunk.end_ms
                if agent.first_audio_at is None:
                    agent.first_audio_at = agent.marks["first_audio"] = time.perf_counter()
                for frame in frames:
                    if agent.cut:
                        break
                    await self._send_bytes_unlocked(frame)
            if index == 0 and not agent.cut:
                await self._set_state("speaking")

    def _schedule_playback_end(self, agent: AgentTurn) -> None:
        """End the turn when the client says playback_done, or soon after its audio should have finished playing."""
        now = time.perf_counter()
        start = agent.first_audio_at or now
        remaining = max(0.0, start + agent.sent_ms / 1000 - now)
        loop = asyncio.get_running_loop()
        agent.playback_timer = loop.call_later(
            remaining + PLAYBACK_GRACE_S, lambda: self._spawn(self._finish_turn(agent), f"{self.id}-playback-end")
        )

    async def _finish_turn(self, agent: AgentTurn) -> None:
        if agent.playback_timer is not None:
            agent.playback_timer.cancel()
        if self._turn is not agent:
            return
        self._turn = None
        if self._pending is not None and self._pending.agent is agent:
            await self._decide(self._pending, "resume")  # the answer finished while deciding: nothing to stop
        await self._set_state("listening")

    def _played_ms(self, agent: AgentTurn, explicit: float | None) -> float:
        """How much of the turn's audio the user heard: the client's number when it gave one, else its last progress
        report plus the time since, else the time since the first audio was sent; never more than was sent."""
        if explicit is not None:
            return min(explicit, agent.sent_ms)
        now = time.perf_counter()
        if agent.reported_ms is not None:
            return min(agent.sent_ms, agent.reported_ms + (now - agent.reported_at) * 1000)
        if agent.first_audio_at is None:
            return 0.0
        return min(agent.sent_ms, (now - agent.first_audio_at) * 1000)

    def _cut(self, agent: AgentTurn) -> None:
        """Mute a turn at once (synchronously, before the decision is announced): from now on none of its sources,
        deltas, audio chunks or frames are sent, and it is no longer the active turn."""
        agent.cut = True
        if agent.playback_timer is not None:
            agent.playback_timer.cancel()
        if self._turn is agent:
            self._turn = None
        pending = self._pending
        if pending is not None and pending.agent is agent:  # e.g. "stop" while a barge-in was being decided
            self._pending = None
            if pending.deadline is not None and pending.deadline is not asyncio.current_task():
                pending.deadline.cancel()

    async def _interrupt(self, agent: AgentTurn, reason: InterruptReason, played_ms: float | None) -> None:
        """Cut an answer short: cancel generation and speech, save what was heard, back to listening.

        The turn's ``agent_message`` (with ``heard_text``; empty text if nothing was generated yet) is sent before
        anything of the next turn: ``_start_turn`` waits for ``_turn_lock``."""
        if agent.interrupted:
            return
        agent.interrupted = True
        self._cut(agent)
        async with self._turn_lock:
            played = self._played_ms(agent, played_ms)
            heard = heard_text(agent.chunks, played)
            agent.stop.heard_text, agent.stop.reason = heard, reason
            await self._set_state("interrupted")
            if agent.task is not None and not agent.task.done():
                agent.task.cancel()
                await asyncio.wait([agent.task])  # the pipeline saves the stopped answer before the task ends
            message = agent.stop.saved
            complete = agent.message or agent.stop.completed
            fully_heard = agent.audio_done and played >= agent.sent_ms
            if message is None and complete is not None and not fully_heard:  # fully heard: already sent, not cut
                message = complete
                try:
                    message = await self.messages.record_interruption(complete.id, heard_text=heard, reason=reason)
                except Exception as e:
                    log.exception("voice session %s: saving the interruption failed", self.id)
                    await self._error("storage", f"could not save the interrupted answer: {_describe(e)}")
            if message is not None:
                if agent.stop.user is not None:
                    await self._announce(agent, agent.stop.user)  # if the cut came before they went out
                await self._send({"type": "agent_message", "message": _message_json(message)})
            log.info(
                "voice session %s: turn %d %s after %.0f ms of audio (heard %d words)",
                self.id,
                agent.id,
                "stopped" if reason == "stop" else "interrupted",
                played,
                len(heard.split()),
            )
        await self._set_state("listening" if self._turn is None else self._current_state())

    # -------------------------------------------------------------- barge-in

    async def _on_barge_in_start(self, message: BargeInStart) -> None:
        agent = self._turn
        if agent is None or agent.id != message.turn_id or agent.interrupted:
            await self._send({"type": "barge_in", "turn_id": message.turn_id, "decision": "resume"})
            return
        if self._pending is not None:
            return  # already deciding
        agent.barge_in_ms = message.played_ms
        pending = PendingBargeIn(agent, message.played_ms)
        self._pending = pending
        pending.deadline = self._spawn(self._barge_in_deadline(pending), f"{self.id}-barge-in-deadline")
        await self._barge_in_check()

    async def _barge_in_deadline(self, pending: PendingBargeIn) -> None:
        await asyncio.sleep(self.barge_in.decision_timeout_ms / 1000)
        await self._evaluate(pending, deadline_passed=True)

    async def _barge_in_check(self) -> None:
        """After new audio: transcribe the interrupting speech once it is long enough, and re-evaluate."""
        pending = self._pending
        if pending is None:
            return
        ep = self._endpointer
        if pending.stt is None and ep.in_utterance and ep.speech_ms >= self.vad_settings.min_speech_ms:
            pending.stt = self._spawn(self._barge_in_transcript(pending, ep.snapshot()), f"{self.id}-barge-in-stt")
        await self._evaluate(pending, deadline_passed=False)

    async def _barge_in_transcript(self, pending: PendingBargeIn, audio: np.ndarray) -> None:
        transcript = await self._transcribe(audio, partial=True)
        if transcript is not None and self._pending is pending:
            pending.transcript = transcript.text
            pending.backchannel = is_backchannel(transcript.text, self.barge_in.backchannel_max_words)
            await self._evaluate(pending, deadline_passed=False)

    async def _evaluate(self, pending: PendingBargeIn, *, deadline_passed: bool) -> None:
        if self._pending is not pending:
            return
        ep = self._endpointer
        evidence = BargeInEvidence(
            speech_ms=ep.speech_ms,
            speaking=ep.speaking,
            transcript=pending.transcript,
            ended=False,
            deadline_passed=deadline_passed,
        )
        verdict = barge_in_verdict(
            evidence, min_speech_ms=self.vad_settings.min_speech_ms, is_backchannel=pending.backchannel
        )
        if verdict is not None:
            await self._decide(pending, verdict)

    async def _decide(self, pending: PendingBargeIn, verdict: Verdict) -> None:
        if self._pending is not pending:
            return
        self._pending = None
        if pending.deadline is not None and pending.deadline is not asyncio.current_task():
            pending.deadline.cancel()
        if verdict == "stop":
            self._cut(pending.agent)  # muted before the decision goes out: nothing of the turn follows it
        await self._send({"type": "barge_in", "turn_id": pending.agent.id, "decision": verdict})
        if verdict == "stop":
            await self._interrupt(pending.agent, "barge_in", pending.played_ms)

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


# ------------------------------------------------------------------ registry


class VoiceSessions:
    """The live voice sessions of this process, at most one per chat: opening a second one closes the first (4409)."""

    def __init__(self, container: Container) -> None:
        self.container = container
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
            session = VoiceSession.from_container(chat, transport, self.container)
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
