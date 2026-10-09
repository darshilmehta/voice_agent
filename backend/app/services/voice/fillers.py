"""Fixed phrases a voice turn says besides its answer. The filler while it searches the web (docs/DESIGN.md §3.7):
"Let me look that up." / "मैं देखता हूँ।", spoken the moment the route asks for live data, so the search starts
without silence. The visual's tail (§12.1): "It's on screen now." / "स्क्रीन पर दिखा दिया है।", after an answer whose
visual became ready while the answer was still being heard.

The audio is synthesized once per TTS provider, phrase and language (the filler at startup when web search is
enabled: the model preloader calls ``preload``; the tail when a voice session's first visual starts, seconds before
it can be needed), else on first use, and then sent like any other ``audio_chunk`` (marked ``filler`` / ``tail``).
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
# Said after an answer when its visual is on screen while the answer is still being heard (§12.1, the spoken tail).
VISUAL_TAILS: dict[Language, str] = {"en": "It's on screen now.", "hi": "स्क्रीन पर दिखा दिया है।"}


class FillerAudio:
    """Fixed phrases' PCM for one TTS provider (the web search filler, the visual's tail), cached per text and
    language."""

    def __init__(self, tts: SpeechSynthesizer) -> None:
        self.tts = tts
        self._pcm: dict[tuple[str, Language], bytes] = {}
        self._lock = asyncio.Lock()

    def text(self, language: Language) -> str:
        return FILLERS[language]

    def cached(self, language: Language) -> bytes | None:
        return self._pcm.get((FILLERS[language], language))

    async def get(self, language: Language) -> bytes:
        """The filler's audio: cached, or synthesized now (and kept)."""
        return await self.phrase(FILLERS[language], language)

    async def phrase(self, text: str, language: Language) -> bytes:
        """A fixed phrase's audio: cached, or synthesized now (and kept)."""
        key = (text, language)
        pcm = self._pcm.get(key)
        if pcm is not None:
            return pcm
        async with self._lock:
            pcm = self._pcm.get(key)
            if pcm is None:
                pcm = self._pcm[key] = await self.tts.synthesize(text, language)
        return pcm

    async def preload(self, languages: Iterable[Language], *, fillers: bool = True, tails: bool = False) -> None:
        for language in languages:
            if fillers and language in FILLERS:
                await self.get(language)
            if tails and language in VISUAL_TAILS:
                await self.phrase(VISUAL_TAILS[language], language)


_AUDIO: WeakKeyDictionary[SpeechSynthesizer, FillerAudio] = WeakKeyDictionary()


def filler_audio(tts: SpeechSynthesizer) -> FillerAudio:
    """The fillers of this TTS provider (one cache per provider instance)."""
    audio = _AUDIO.get(tts)
    if audio is None:
        audio = _AUDIO[tts] = FillerAudio(tts)
    return audio
