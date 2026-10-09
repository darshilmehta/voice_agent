"""Whisper with a vocabulary prompt (providers/speech.py, DESIGN §9.3): the prompt of the spoken language, one bounded
greedy pass, the unprompted decode when the prompted one can't be trusted, and the confidence on the transcript.
The model itself is replaced: these tests run without the ml group."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

from app.providers.registry import build_container
from app.providers.speech import (
    PROMPT_MAX_TOKENS,
    SAMPLE_RATE,
    Transcript,
    echoes_prompt,
    prompted_decode_failed,
    prompted_token_cap,
    segment_confidence,
)

from .conftest import make_model_assets

PROMPTS = {"en": "Valmora Industries, Zephyra Logistics, EBITDA.", "hi": "वाल्मोरा, Valmora के बारे में।"}


def seg(logprob: float = -0.2, no_speech: float = 0.01, ratio: float = 1.1, tokens: int = 10) -> dict[str, Any]:
    return {"avg_logprob": logprob, "no_speech_prob": no_speech, "compression_ratio": ratio, "tokens": [0] * tokens}


class Scripted:
    """Stands in for the model: the language it detects and what each decode (with or without a prompt) gives."""

    def __init__(self, language: str, prompted: tuple[str, list], plain: tuple[str, list]) -> None:
        self.language = language
        self.prompted = prompted
        self.plain = plain
        self.decodes: list[str | None] = []

    def install(self, stt: Any) -> None:
        stt._model_locked = lambda: object()
        stt._language_locked = lambda model, audio, languages: (
            (self.language, 0.9) if len(languages) > 1 else (languages[0], None)
        )

        def decode(model: Any, audio: np.ndarray, language: str, prompt: str | None) -> tuple[str, list]:
            self.decodes.append(prompt)
            return self.prompted if prompt is not None else self.plain

        stt._decode_locked = decode


@pytest.fixture
def stt(load_local, tmp_path):
    make_model_assets(tmp_path)
    return build_container(load_local())["stt"]


def transcribe(stt: Any, prompts: dict[str, str] | None = PROMPTS, languages=("en", "hi")) -> Transcript:
    audio = np.full(SAMPLE_RATE, 0.1, dtype=np.float32)
    return asyncio.run(stt.transcribe(audio, list(languages), prompts=prompts))


def test_the_spoken_languages_prompt_is_used(stt):
    model = Scripted("hi", ("वाल्मोरा का revenue कितना था?", [seg(-0.3, 0.05)]), ("x", [seg()]))
    model.install(stt)
    t = transcribe(stt)
    assert model.decodes == [PROMPTS["hi"]]
    assert (t.text, t.language, t.prompted) == ("वाल्मोरा का revenue कितना था?", "hi", True)
    assert (t.avg_logprob, t.no_speech_prob) == (-0.3, 0.05)
    assert not t.likely_misheard


def test_without_prompts_one_plain_decode(stt):
    model = Scripted("en", ("never", []), ("What was the revenue?", [seg(-0.25)]))
    model.install(stt)
    t = transcribe(stt, prompts=None)
    assert model.decodes == [None] and t.text == "What was the revenue?" and not t.prompted


def test_a_language_without_a_prompt_decodes_plainly(stt):
    model = Scripted("en", ("never", []), ("Okay.", [seg()]))
    model.install(stt)
    assert transcribe(stt, prompts={"hi": "वाल्मोरा"}).prompted is False
    assert model.decodes == [None]


@pytest.mark.parametrize(
    ("text", "segments"),
    [
        ("ॐ ॐ ॐ ॐ ॐ ॐ ॐ ॐ", [seg(-0.08, ratio=17.4, tokens=224)]),  # a repetition loop (measured on Hindi clips)
        ("Thank you.", [seg(-0.81, no_speech=0.9)]),  # noise, with a prompt
        ("Valmora Industries, Zephyra Logistics", [seg(-0.3)]),  # the prompt, echoed
        ("Something garbled", [seg(-1.4)]),  # a failed decode
    ],
)
def test_an_untrustworthy_prompted_decode_is_redone_without_the_prompt(stt, text, segments):
    model = Scripted("en", (text, segments), ("", []))
    model.install(stt)
    t = transcribe(stt)
    assert model.decodes == [PROMPTS["en"], None]
    assert (t.text, t.prompted, t.avg_logprob, t.no_speech_prob) == ("", False, None, None)


def test_confidence_is_token_weighted_and_takes_the_highest_no_speech():
    assert segment_confidence([seg(-0.2, 0.1, tokens=30), seg(-1.0, 0.4, tokens=10)]) == (pytest.approx(-0.4), 0.4)
    assert segment_confidence([]) == (None, None)


@pytest.mark.parametrize(
    ("transcript", "misheard"),
    [
        (Transcript("What was the revenue?", "en", avg_logprob=-0.25, no_speech_prob=0.02), False),
        (Transcript("Vamar aur jephir mean kiske", "hi", avg_logprob=-0.91, no_speech_prob=0.06), True),
        (Transcript("Thank you.", "en", avg_logprob=-0.4, no_speech_prob=0.9), True),
        (Transcript("", "en", avg_logprob=-2.0), False),  # nothing to repeat
        (Transcript("Okay.", "en"), False),  # a recognizer without confidence
    ],
)
def test_likely_misheard(transcript, misheard):
    assert transcript.likely_misheard is misheard


def test_echo_needs_two_words_of_the_prompt():
    assert echoes_prompt("Valmora Industries.", "Valmora Industries, Zephyra Logistics.")
    assert not echoes_prompt("Valmora.", "Valmora Industries, Zephyra Logistics.")
    assert not echoes_prompt("What was Valmora's revenue?", "Valmora Industries, Zephyra Logistics.")
    assert not echoes_prompt("Valmora Industries", None)


def test_prompted_decode_checks():
    assert not prompted_decode_failed("What was Valmora's revenue?", [seg()], "Valmora Industries")
    assert prompted_decode_failed("x", [seg(ratio=2.5)], "p")


def test_the_prompted_decode_is_capped_by_the_audio_length():
    assert prompted_token_cap(SAMPLE_RATE * 2) == 76  # 2 s of speech: at most 76 tokens, not 224
    assert prompted_token_cap(SAMPLE_RATE * 30) == PROMPT_MAX_TOKENS


def test_mlx_counts_tokens_without_the_model(stt):
    pytest.importorskip("mlx_whisper")
    assert stt.count_tokens(" revenue") == 1
    assert stt.count_tokens(" Valmora") == 3
