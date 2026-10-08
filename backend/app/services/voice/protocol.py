"""The voice WebSocket protocol (``WS /ws/chats/{chat_id}/voice``): message shapes, audio frames, close codes.

Binary frames
    client → server  PCM16 LE mono 16 kHz, 20-64 ms per frame, sent continuously while the mic is on
    server → client  12-byte header (uint32 LE turn_id, chunk_index, seq) + PCM16 LE mono 24 kHz;
                     ``seq`` counts the frames of one chunk from 0; play only the current turn's frames

JSON text frames (field ``type``)
    client → server  start {language} · barge_in_start {turn_id, played_ms} · playback {turn_id, played_ms} ·
                     playback_done {turn_id} · stop {} · end {}
    server → client  ready · state · user_speech · transcript_partial · user_message · turn · sources · delta ·
                     audio_chunk · agent_message · barge_in · error

Close codes: 1000 end, 1001 server shutdown, 1011 internal error, 4403 origin not allowed (browser pages from origins
outside server.cors_allowed_origins), 4404 unknown chat, 4409 replaced by a newer session for the chat.
"""

from __future__ import annotations

import json
import struct
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from ...settings import Language

INPUT_SAMPLE_RATE = 16_000
OUTPUT_FRAME_MS = 200  # audio per server → client binary frame
MAX_INPUT_FRAME_BYTES = INPUT_SAMPLE_RATE * 2  # 1 s of PCM16; clients send 20-64 ms
MAX_MESSAGE_BYTES = 64 * 1024  # any WebSocket message (uvicorn ws_max_size); larger ones close the connection
HEADER = struct.Struct("<III")  # turn_id, chunk_index, seq

CLOSE_NORMAL = 1000
CLOSE_GOING_AWAY = 1001  # server shutting down
CLOSE_INTERNAL_ERROR = 1011
CLOSE_FORBIDDEN_ORIGIN = 4403  # a browser page from an origin not in server.cors_allowed_origins
CLOSE_CHAT_NOT_FOUND = 4404
CLOSE_REPLACED = 4409  # another voice session opened for the same chat

AgentState = Literal["listening", "thinking", "speaking", "interrupted"]
ErrorStage = Literal["stt", "retrieval", "llm", "tts", "storage", "audio"]


class _Message(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class Start(_Message):
    type: Literal["start"]
    language: Language | None = None  # None: detect per utterance among stt.languages


class BargeInStart(_Message):
    type: Literal["barge_in_start"]
    turn_id: int = Field(ge=0)
    played_ms: float = Field(ge=0)


class Playback(_Message):
    type: Literal["playback"]
    turn_id: int = Field(ge=0)
    played_ms: float = Field(ge=0)


class PlaybackDone(_Message):
    type: Literal["playback_done"]
    turn_id: int = Field(ge=0)


class Stop(_Message):
    type: Literal["stop"]


class End(_Message):
    type: Literal["end"]


ClientMessage = Annotated[Start | BargeInStart | Playback | PlaybackDone | Stop | End, Field(discriminator="type")]
_CLIENT = TypeAdapter(ClientMessage)


class ProtocolError(ValueError):
    """A client message that can't be understood."""


def parse_client_message(text: str) -> ClientMessage:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"control messages must be JSON: {e.msg}") from None
    try:
        return _CLIENT.validate_python(data)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(str(p) for p in err['loc']) or 'message'}: {err['msg']}" for err in e.errors())
        raise ProtocolError(f"invalid control message: {problems}") from None


def dumps(message: dict[str, Any]) -> str:
    return json.dumps(message, ensure_ascii=False, separators=(",", ":"))


def audio_frames(turn_id: int, chunk_index: int, pcm: bytes, *, sample_rate: int) -> list[bytes]:
    """A chunk's PCM16 audio as binary frames of OUTPUT_FRAME_MS each (the last may be shorter), headers included."""
    size = max(2, sample_rate * OUTPUT_FRAME_MS // 1000 * 2)
    return [
        HEADER.pack(turn_id, chunk_index, seq) + pcm[start : start + size]
        for seq, start in enumerate(range(0, len(pcm), size))
    ]


def parse_frame(frame: bytes) -> tuple[int, int, int, bytes]:
    """(turn_id, chunk_index, seq, pcm) of a server → client frame (clients and tests)."""
    turn_id, chunk_index, seq = HEADER.unpack_from(frame)
    return turn_id, chunk_index, seq, frame[HEADER.size :]
