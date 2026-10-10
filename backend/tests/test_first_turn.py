"""Polish round, item 5: the first spoken question of a new chat (14.2 s to first audio against a 4.7 s median). A
voice session opening asks the LLM nothing that question would queue behind (the names' Devanagari spellings are
asked for after the preload, when a document becomes READY, or after the session's first turn), and warms the
conversation models again (their tiny inference: weights paged out while idle come back; the LLM, if Ollama unloaded
it, loaded with the router's and the answer's prompts)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from app.providers.llm import LLMClient
from app.services import preload
from app.services.voice import vocabulary as vocab
from app.services.voice.vocabulary import SpeechVocabulary, VocabularyWarmer

from .test_speech_vocabulary import Chat, Docs, JsonLLM, count


@pytest.fixture(autouse=True)
def _fresh_transliterations(monkeypatch):
    monkeypatch.setattr(vocab, "_TRANSLITERATIONS", {})


async def test_a_session_opening_asks_the_model_nothing_and_its_first_turn_completes_the_names():
    llm = JsonLLM({"names": ["वाल्मोरा"]})
    sv = SpeechVocabulary(Docs(), None, count, llm)  # type: ignore[arg-type]
    opening = await sv.hints(Chat(), use_model=False)  # type: ignore[arg-type]
    assert llm.calls == [] and "वाल्मोरा" not in opening.prompts["hi"]
    assert "Valmora" in opening.prompts["hi"] and "Valmora" in opening.prompts["en"]
    assert await sv.hints(Chat(), use_model=False) is opening  # (cached: still nothing asked)
    after_turn = await sv.hints(Chat())  # type: ignore[arg-type]  # the refresh after the first turn
    assert llm.calls == [["Valmora"]] and after_turn.prompts["hi"].startswith("वाल्मोरा, Valmora")
    assert await sv.hints(Chat()) is after_turn and len(llm.calls) == 1  # type: ignore[arg-type]


async def test_names_spelled_ahead_need_no_model_call_when_a_session_opens():
    llm = JsonLLM({"names": ["वाल्मोरा"]})
    sv = SpeechVocabulary(Docs(), None, count, llm)  # type: ignore[arg-type]
    warmer = VocabularyWarmer(sv, projects=lambda: _async(["prj_1"]))
    warmer.warm_all()
    await asyncio.gather(*warmer._tasks)
    assert llm.calls == [["Valmora"]]
    hints = await sv.hints(Chat(), use_model=False)  # type: ignore[arg-type]
    assert hints.prompts["hi"].startswith("वाल्मोरा, Valmora") and len(llm.calls) == 1
    await warmer.document_ready("prj_1", "doc_v", 1)  # a document READY: spelled in the background
    await asyncio.gather(*warmer._tasks)
    await warmer.close()


async def _async(value: Any) -> Any:
    return value


class FakeLLM(LLMClient):
    def __init__(self, loaded: bool | None) -> None:  # (no provider config needed here)
        self._loaded = loaded
        self.preloads = 0
        self.warmed: list[str] = []

    async def loaded(self) -> bool | None:
        return self._loaded

    async def preload(self) -> None:
        self.preloads += 1

    async def warm_up(self, prompts, *, model=None) -> None:
        self.warmed += [m[0].content[:30] for m in prompts]


class Touchable:
    def __init__(self, loaded: bool = True) -> None:
        self.loaded, self.touched = loaded, 0

    async def touch(self) -> bool:
        self.touched += 1
        return self.loaded


def container(llm: Any, **providers: Any) -> Any:
    settings = SimpleNamespace(
        llm=SimpleNamespace(router_model="qwen3:4b-instruct", chat_model="qwen3:4b-instruct"),
        client=SimpleNamespace(languages=["en", "hi"], default_language="en"),
    )
    return SimpleNamespace(settings=settings, providers={"llm": llm, **providers})


async def test_rewarm_touches_the_local_models_and_reloads_an_unloaded_llm():
    stt, emb, rr, tts = Touchable(), Touchable(), Touchable(loaded=False), Touchable()
    llm = FakeLLM(loaded=False)
    done = await preload.rewarm(container(llm, stt=stt, embeddings=emb, reranker=rr, tts=tts), "hi")
    assert (stt.touched, emb.touched, rr.touched, tts.touched) == (1, 1, 1, 1)
    assert set(done) >= {"stt", "embeddings", "tts", "llm_loaded", "llm_reloaded_s"} and "reranker" not in done
    assert llm.preloads == 1 and len(llm.warmed) == 2  # the router's prompt, then the Hindi answer's

    loaded = FakeLLM(loaded=True)
    done = await preload.rewarm(container(loaded, stt=Touchable()))
    assert loaded.preloads == 0 and loaded.warmed == [] and done["llm_loaded"] is True


async def test_rewarm_never_raises_for_a_failing_model():
    class Broken(Touchable):
        async def touch(self) -> bool:
            raise RuntimeError("metal said no")

    done = await preload.rewarm(container(FakeLLM(None), stt=Broken(), embeddings=Touchable()))
    assert done["errors"] == ["RuntimeError: metal said no"]


def test_a_voice_session_opening_warms_the_models_at_most_once_a_minute(make_app, fakes, monkeypatch):
    calls: list[float] = []

    async def fake_rewarm(container, language=None):
        calls.append(1)
        return {}

    monkeypatch.setattr(preload, "rewarm", fake_rewarm)
    with make_app() as api:
        project = api.post("/api/projects", json={"name": "P"}).json()["id"]
        chat = api.post(f"/api/projects/{project}/chats", json={}).json()["id"]
        for _ in range(2):
            with api.websocket_connect(f"/ws/chats/{chat}/voice") as ws:
                ws.send_json({"type": "start", "language": None})
                assert ws.receive_json()["type"] == "ready"
                ws.send_json({"type": "end"})
        assert calls == [1]  # the second session, a moment later, doesn't warm again


async def test_touch_reruns_the_warm_up_and_never_loads_a_model():
    import threading

    from app.providers.models import LazyModelProvider

    class Model(LazyModelProvider):
        uses_torch = False

        def __init__(self) -> None:  # (no config: nothing here reads it)
            self._lock, self._model, self._load_error, self.warms = threading.Lock(), None, None, 0

        def _warm(self, model: Any) -> None:
            self.warms += 1

    m = Model()
    assert await m.touch() is False and m.warms == 0  # not loaded: left alone
    m._model = object()
    assert await m.touch() is True and m.warms == 1
