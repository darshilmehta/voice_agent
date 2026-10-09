"""The voice loop (docs/DESIGN.md §3.3): a live voice session per chat on top of the chat answer pipeline.

protocol.py      the WebSocket protocol: messages, audio frames, close codes
turn_taking.py   endpointing on the server VAD and the barge-in verdict (pure)
speech_text.py   speakable chunks, heard text, backchannels (pure)
session.py       VoiceSession (state machine per connection) and VoiceSessions (one per chat)
"""

from .session import Transport, VoiceSession, VoiceSessions

__all__ = ["Transport", "VoiceSession", "VoiceSessions"]
