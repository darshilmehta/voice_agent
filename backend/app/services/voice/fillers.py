"""The filler a voice turn says while it searches the web (docs/DESIGN.md §3.7): "Let me look that up." /
"मैं देखता हूँ।", spoken the moment the route asks for live data, so the search starts without silence.

The audio is synthesized once per TTS provider and language, at startup when web search is enabled (the model
preloader calls ``preload``), else on first use, and then sent like any other ``audio_chunk`` (marked ``filler``).
It costs no TTS time per turn.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from weakref import WeakKeyDictionary

from ...providers.speech import SpeechSynthesizer
from ...settings import Language

log = logging.getLogger(__name__)

FILLERS: dict[Language, str] = {"en": "Let me look that up.", "hi": "मैं देखता हूँ।"}


class FillerAudio:
    """The fillers' PCM for one TTS provider, cached per language."""

    def __init__(self, tts: SpeechSynthesizer) -> None:
        self.tts = tts
        self._pcm: dict[Language, bytes] = {}
        self._lock = asyncio.Lock()

    def text(self, language: Language) -> str:
        return FILLERS[language]

    def cached(self, language: Language) -> bytes | None:
        return self._pcm.get(language)

    async def get(self, language: Language) -> bytes:
        """The filler's audio: cached, or synthesized now (and kept)."""
        pcm = self._pcm.get(language)
        if pcm is not None:
            return pcm
        async with self._lock:
            pcm = self._pcm.get(language)
            if pcm is None:
                pcm = self._pcm[language] = await self.tts.synthesize(FILLERS[language], language)
        return pcm

    async def preload(self, languages: Iterable[Language]) -> None:
        for language in languages:
            if language in FILLERS:
                await self.get(language)


_AUDIO: WeakKeyDictionary[SpeechSynthesizer, FillerAudio] = WeakKeyDictionary()


def filler_audio(tts: SpeechSynthesizer) -> FillerAudio:
    """The fillers of this TTS provider (one cache per provider instance)."""
    audio = _AUDIO.get(tts)
    if audio is None:
        audio = _AUDIO[tts] = FillerAudio(tts)
    return audio
