"""Phases 4-6 acceptance, end to end over the voice WebSocket with the real stack: Silero → mlx-whisper (or
faster-whisper) → BGE-M3 + Qdrant + reranker → Ollama → Kokoro.

    RUN_INTEGRATION=1 uv run --group ml pytest tests/integration/test_voice_e2e.py -s

Needs, besides the models, the smoke documents and Qdrant (see conftest): Ollama with qwen3:4b-instruct, and the
smoke-test speech clips in SMOKE_AUDIO (see conftest; written by scripts/smoke/09_kokoro.py).

A spoken question about revenue is streamed in real time (then silence) and must come back as a transcript, a cited
spoken answer and audio frames. Mid-answer the user barges in with a correction: the answer stops, what was heard is
saved, the correction is answered (and Whisper must understand that spoken answer again). A Hindi question checks
language detection. Latency from the end of the user's speech to each stage is printed in the summary (DESIGN §9.5).
"""

from __future__ import annotations

import contextlib
import json
import queue
import statistics
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.providers.registry import build_container
from app.services.voice.protocol import parse_frame

from .conftest import METRICS, integration_settings, require_chat_model
from .test_chat_e2e import upload_ready

pytestmark = pytest.mark.integration

FRAME = 512  # 32 ms at 16 kHz, sent in real time
PRELOAD_TIMEOUT_S = 300
TURN_TIMEOUT_S = 90

Check = Callable[[Any], bool]


def pcm16k(path: Path) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != 16_000:
        g = np.gcd(rate, 16_000)
        audio = resample_poly(audio, 16_000 // g, rate // g).astype(np.float32)
    return audio


def speech_end_s(audio: np.ndarray) -> float:
    """Where the clip's speech ends (its own trailing silence excluded): 10 ms envelope above 2% of its peak."""
    env = np.convolve(np.abs(audio), np.ones(160) / 160, mode="same")
    voiced = np.flatnonzero(env > 0.02 * float(np.abs(audio).max()))
    return (voiced[-1] + 1) / 16_000


def is_type(type_: str, **fields: Any) -> Check:
    return lambda m: isinstance(m, dict) and m.get("type") == type_ and all(m.get(k) == v for k, v in fields.items())


def frame_of(turn_id: int) -> Check:
    return lambda m: isinstance(m, bytes) and parse_frame(m)[0] == turn_id


@dataclass
class Received:
    at: float  # perf_counter when it arrived
    item: Any  # dict (JSON) or bytes (audio frame)


@dataclass
class LiveClient:
    """Streams audio in real time while a reader thread timestamps everything the server sends."""

    ws: Any
    inbox: queue.Queue[Received] = field(default_factory=queue.Queue)
    log: list[Received] = field(default_factory=list)

    def __post_init__(self) -> None:
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        while True:
            try:
                message = self.ws.receive()
            except Exception:
                return
            now = time.perf_counter()
            if message["type"] == "websocket.close":
                self.inbox.put(Received(now, {"type": "closed", "code": message.get("code")}))
                return
            item = json.loads(message["text"]) if message.get("text") is not None else message.get("bytes")
            self.inbox.put(Received(now, item))

    def drain(self) -> list[Received]:
        while True:
            try:
                self.log.append(self.inbox.get_nowait())
            except queue.Empty:
                return self.log

    def wait(self, check: Check, *, since: float = 0.0, timeout: float = TURN_TIMEOUT_S) -> Received:
        """The first message that arrived at or after ``since`` and matches, waiting for it if needed."""
        deadline = time.monotonic() + timeout
        while True:
            found = next((r for r in self.drain() if r.at >= since and check(r.item)), None)
            if found is not None:
                return found
            if time.monotonic() > deadline:
                raise TimeoutError(f"no matching message within {timeout}s")
            with contextlib.suppress(queue.Empty):
                self.log.append(self.inbox.get(timeout=0.05))

    def send(self, type_: str, **fields: Any) -> None:
        self.ws.send_text(json.dumps({"type": type_, **fields}))

    def stream(self, audio: np.ndarray, *, then_silence_s: float, stop_silence_on: Check | None = None) -> float:
        """Send audio (float32 16 kHz) as PCM16 frames in real time, then silence (cut short once a message matching
        ``stop_silence_on`` has arrived). Returns the wall time at which the clip's speech ended."""
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
        frames = np.concatenate([pcm, np.zeros(int(then_silence_s * 16_000), dtype="<i2")])
        start = time.perf_counter()
        for i in range(0, len(frames), FRAME):
            if (pause := start + i / 16_000 - time.perf_counter()) > 0:
                time.sleep(pause)
            self.ws.send_bytes(frames[i : i + FRAME].tobytes())
            silence = i >= len(pcm)
            if silence and stop_silence_on and any(r.at >= start and stop_silence_on(r.item) for r in self.drain()):
                break
        return start + speech_end_s(audio)


def after(log: list[Received], t: float) -> list[Received]:
    return [r for r in log if r.at >= t]


def first(log: list[Received], check: Check) -> Received:
    return next(r for r in log if check(r.item))


def latency_row(log: list[Received], speech_end: float, turn_id: int, user: dict, agent: dict) -> dict[str, Any]:
    """Client-side ms from the end of the user's speech to each stage, plus the server's stage timings."""

    def ms(check: Check) -> int | None:
        found = next((r for r in after(log, speech_end) if check(r.item)), None)
        return round((found.at - speech_end) * 1000) if found else None

    return {
        "end_of_turn": ms(is_type("user_speech", phase="end")),
        "user_message": ms(is_type("user_message")),
        "first_delta": ms(is_type("delta", turn_id=turn_id)),
        "first_audio": ms(frame_of(turn_id)),
        "server: stt_ms (after end of turn)": user["latency"]["stt_ms"],
        "server: speculative_stt": user["latency"]["speculative_stt"],
        "server: retrieval_ms": agent["latency"]["retrieval_ms"],
        "server: rerank_ms": agent["latency"]["rerank_ms"],
        "server: first_delta_ms (from turn start)": agent["latency"]["first_delta_ms"],
    }


@pytest.fixture(scope="module")
def api(models_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    settings = integration_settings(models_root, tmp_path_factory.mktemp("voice"))
    require_chat_model(settings)
    container = build_container(settings)
    with TestClient(create_app(settings, container)) as client:  # preloads the models in the background
        t0 = time.perf_counter()
        while (report := client.get("/health").json()["preload"])["state"] == "loading":
            assert time.perf_counter() - t0 < PRELOAD_TIMEOUT_S, report
            time.sleep(0.5)
        loads = (f"{m['capability']} {m['state']} {m['seconds']}s" for m in report["models"])
        METRICS["voice preload"] = ", ".join(loads)
        assert report["state"] == "ready", report
        try:
            yield client
        finally:
            client.portal.call(container["vector_store"].drop_collection)  # type: ignore[attr-defined]


def test_spoken_question_barge_in_correction_and_hindi(api, smoke_docs, smoke_audio):
    project = api.post("/api/projects", json={"name": "Annual report FY24"}).json()["id"]
    upload_ready(api, project, smoke_docs / "annual_report.pdf", "annual_report.pdf (voice)")
    chat = api.post(f"/api/projects/{project}/chats", json={}).json()["id"]
    stt = api.app.state.container["stt"]  # type: ignore[attr-defined]
    rows: dict[str, dict[str, Any]] = {}

    with api.websocket_connect(f"/ws/chats/{chat}/voice") as ws:
        c = LiveClient(ws)
        c.send("start", language=None)  # detect the language of every utterance
        assert c.wait(is_type("ready")).item["output_sample_rate"] == 24_000

        # ---- turn 1: a spoken question, answered in speech
        end1 = c.stream(pcm16k(smoke_audio / "say_en_0.wav"), then_silence_s=2.0, stop_silence_on=frame_of(1))
        first_audio = c.wait(frame_of(1))
        user1 = first(c.drain(), is_type("user_message")).item["message"]
        assert "revenue" in user1["text"].lower(), user1
        assert (user1["language"], user1["modality"]) == ("en", "voice")
        assert first(c.log, is_type("turn")).item == {"type": "turn", "turn_id": 1}
        sources = first(c.log, is_type("sources")).item
        assert not sources["abstained"] and sources["sources"], sources

        # ---- the user hears ~1.5 s of the answer, then barges in with a correction
        time.sleep(max(0.0, first_audio.at + 1.5 - time.perf_counter()))
        c.send("barge_in_start", turn_id=1, played_ms=(time.perf_counter() - first_audio.at) * 1000)
        barge_in_sent = time.perf_counter()
        end2 = c.stream(pcm16k(smoke_audio / "say_en_1.wav"), then_silence_s=2.5, stop_silence_on=is_type("turn"))
        decision = c.wait(is_type("barge_in", turn_id=1), since=barge_in_sent)
        assert decision.item["decision"] == "stop"
        stopped = c.wait(lambda m: is_type("agent_message")(m) and m["message"]["heard_text"] is not None)
        agent1 = stopped.item["message"]
        spoken1 = " ".join(r.item["text"] for r in c.log if is_type("audio_chunk", turn_id=1)(r.item))
        assert agent1["heard_text"] and spoken1.startswith(agent1["heard_text"]), (agent1, spoken1)
        assert agent1["route"]["stopped"] and agent1["route"]["interrupted"] == "barge_in"
        rows["turn 1: revenue question"] = latency_row(c.log, end1, 1, user1, agent1)

        # ---- turn 2: the correction becomes the next turn and is answered completely
        done2 = c.wait(lambda m: is_type("agent_message")(m) and m["message"]["seq"] > agent1["seq"])
        log = after(c.drain(), decision.at)
        assert not [r for r in log if frame_of(1)(r.item) or is_type("delta", turn_id=1)(r.item)]
        assert log.index(stopped) < log.index(first(log, is_type("user_message")))  # cut turn closed first
        user2 = first(log, is_type("user_message")).item["message"]
        agent2 = done2.item["message"]
        assert "ebitda" in user2["text"].lower(), user2
        assert "18.2" in agent2["text"] and agent2["citations"], agent2
        rows["turn 2: correction after barge-in"] = latency_row(log, end2, 2, user2, agent2)
        rows["turn 2: correction after barge-in"]["barge_in_decision (after barge_in_start)"] = round(
            (decision.at - barge_in_sent) * 1000
        )
        c.send("playback_done", turn_id=2)
        c.wait(is_type("state", state="listening"), since=done2.at)

        # the spoken answer is real speech: Whisper hears its words again
        pcm2 = b"".join(parse_frame(r.item)[3] for r in log if frame_of(2)(r.item))
        audio2 = np.frombuffer(pcm2, dtype="<i2").astype(np.float32) / 32768
        from scipy.signal import resample_poly

        back = api.portal.call(stt.transcribe, resample_poly(audio2, 2, 3).astype(np.float32), ["en"])
        spoken2 = " ".join(r.item["text"] for r in log if is_type("audio_chunk", turn_id=2)(r.item))
        words = lambda s: {w.strip(".,%?!").lower() for w in s.split()}  # noqa: E731
        assert len(words(back.text) & words(spoken2)) >= 3, (back.text, spoken2)

        # ---- turn 3: a Hindi question; the language is detected
        end3 = c.stream(pcm16k(smoke_audio / "say_hi_0.wav"), then_silence_s=2.0, stop_silence_on=is_type("turn"))
        done3 = c.wait(lambda m: is_type("agent_message")(m) and m["message"]["seq"] > agent2["seq"] + 1)
        log = after(c.drain(), end3)
        user3, agent3 = first(log, is_type("user_message")).item["message"], done3.item["message"]
        assert (user3["language"], agent3["language"]) == ("hi", "hi"), (user3, agent3)
        assert any(frame_of(3)(r.item) for r in log)
        rows["turn 3: Hindi question"] = latency_row(log, end3, 3, user3, agent3)
        c.send("playback_done", turn_id=3)
        c.send("end")

    for label, row in rows.items():
        METRICS[f"voice latency, {label} (ms after end of speech)"] = row
    METRICS["voice first audio, median over turns (ms after end of speech)"] = statistics.median(
        r["first_audio"] for r in rows.values()
    )
    METRICS["voice transcripts"] = [user1["text"], user2["text"], user3["text"]]
    METRICS["voice answers"] = {
        "turn 1 heard before barge-in": agent1["heard_text"],
        "turn 2": agent2["text"],
        "turn 2 audio re-transcribed": back.text,
        "turn 3": agent3["text"],
    }
