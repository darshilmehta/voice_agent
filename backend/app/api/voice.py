"""WS /ws/chats/{chat_id}/voice: a live voice conversation in a chat (docs/DESIGN.md §3.3).

The session itself is ``VoiceSession`` (``app/services/voice``); this endpoint only adapts Starlette's WebSocket to
the session's transport and handles the close codes: a page from an origin outside server.cors_allowed_origins → 4403,
unknown chat → 4404, replaced by a newer session for the same chat → 4409. The connection is accepted before closing
so that browsers see those codes (a refused handshake is reported to scripts as 1006 without a code). Protocol:
``app/services/voice/protocol.py`` and the backend README.
"""

from __future__ import annotations

import contextlib

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from ..services.base import NotFound
from ..services.voice import VoiceSessions
from ..services.voice.protocol import CLOSE_CHAT_NOT_FOUND, CLOSE_FORBIDDEN_ORIGIN

router = APIRouter(tags=["voice"])


class StarletteTransport:
    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket

    async def receive(self) -> bytes | str | None:
        try:
            message = await self.websocket.receive()
        except (WebSocketDisconnect, RuntimeError):
            return None
        if message["type"] == "websocket.disconnect":
            return None
        if message.get("bytes") is not None:
            return message["bytes"]
        return message.get("text") or ""

    async def send_text(self, text: str) -> None:
        await self.websocket.send_text(text)

    async def send_bytes(self, data: bytes) -> None:
        await self.websocket.send_bytes(data)

    async def close(self, code: int, reason: str = "") -> None:
        ws = self.websocket
        if ws.application_state == WebSocketState.CONNECTED and ws.client_state == WebSocketState.CONNECTED:
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await ws.close(code, reason)


def origin_allowed(origin: str | None, allowed: list[str]) -> bool:
    """The CORS middleware doesn't cover WebSockets, so the endpoint checks the Origin itself. Browsers always send
    one on a WebSocket handshake, so a page on another site can't open a session; a request without an Origin is not
    from a browser page (a local script, a test client) and could reach the HTTP API just as well, so it is allowed."""
    return origin is None or "*" in allowed or origin in allowed


@router.websocket("/ws/chats/{chat_id}/voice")
async def voice_session(websocket: WebSocket, chat_id: str) -> None:
    origin = websocket.headers.get("origin")
    allowed_origin = origin_allowed(origin, websocket.app.state.container.settings.server.cors_allowed_origins)
    await websocket.accept()  # accepted before closing, so that browsers see the close code
    transport = StarletteTransport(websocket)
    if not allowed_origin:
        await transport.close(CLOSE_FORBIDDEN_ORIGIN, "origin not allowed")
        return
    sessions: VoiceSessions = websocket.app.state.voice_sessions
    try:
        session = await sessions.open(chat_id, transport)
    except NotFound:
        await transport.close(CLOSE_CHAT_NOT_FOUND, "chat not found")
        return
    try:
        await session.run()
    finally:
        sessions.release(session)
