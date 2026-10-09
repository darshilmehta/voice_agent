"""The canvas in a voice session (docs/DESIGN.md §12.1, §3.10), fake speech: the answer's visual arrives as `visual` and
`canvas` messages with the turn's id, never ahead of the first audio, possibly after `agent_message`; the spoken tail
("It's on screen now.") only once the visual is ready and while the answer is still being heard; barge-in, a new
question or the stop button while a visual is being planned; spoken canvas edits (English and Hindi)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from app.services.voice.fillers import VISUAL_TAILS
from app.services.voice.protocol import parse_frame

from .canvas_turn_helpers import Script, choose, line_of_revenue, planner_calls, report_chat
from .test_voice_api import Item, Voice, VoiceClient, kinds, of, one

SHOW, NEXT, EDIT, HINDI_EDIT = 30, 40, 50, 60  # tones: what the user "says"
REPLY = "Revenue rose in every quarter of FY24 [S1]. It ended the year at its highest level."
TAIL = VISUAL_TAILS["en"]


@pytest.fixture
def voice(make_app, fakes) -> Iterator[Voice]:
    fakes.llm.reply = REPLY
    fakes.stt.scripts.update(
        {
            SHOW: "Show me revenue by quarter",
            NEXT: "Who is the chairperson of the board?",
            EDIT: "Make it a bar chart.",
            HINDI_EDIT: "इसे टेबल में दिखाओ",
        }
    )
    with make_app() as api:
        _, chat_id, _ = report_chat(api)
        yield Voice(api, fakes, chat_id)


def visual_messages(items: list[Item]) -> list[tuple[str, str | None, Any]]:
    return [
        (m["type"], m.get("phase"), m.get("turn_id"))
        for m in items
        if isinstance(m, dict) and m["type"] in ("visual", "canvas")
    ]


def tail_chunks(items: list[Item]) -> list[dict[str, Any]]:
    return [m for m in of(items, "audio_chunk") if m.get("tail")]


def test_the_visual_arrives_after_the_first_audio_and_the_tail_is_spoken_once_it_is_ready(voice):
    script = Script(planner=choose("line", "Revenue")).install(voice.fakes.llm)
    script.planner_delay = 0.3
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(SHOW)
        items = c.until(lambda m: isinstance(m, dict) and m.get("tail") is True)
        items += c.quiet(0.3)
        main = [m for m in items if not (isinstance(m, dict) and m["type"] == "transcript_partial")]
        first_audio = main.index(of(items, "audio_chunk")[0])
        preparing = main.index(next(m for m in of(items, "visual") if m["phase"] == "preparing"))
        assert first_audio < preparing  # the planner starts once the answer's text is complete
        assert visual_messages(items) == [("visual", "preparing", 1), ("visual", "ready", 1), ("canvas", None, 1)]
        ready = next(m for m in of(items, "visual") if m["phase"] == "ready")
        (tail,) = tail_chunks(items)
        assert main.index(ready) < main.index(tail)  # never before the visual is on screen
        answer_chunks = [m for m in of(items, "audio_chunk") if not m.get("tail")]
        assert tail["chunk_index"] == answer_chunks[-1]["chunk_index"] + 1 and tail["text"] == TAIL
        assert tail["turn_id"] == 1
        # here the answer's audio was all sent before the visual was ready: the tail follows agent_message, while
        # the answer is still playing (§3.10's exception), with its frames
        assert main.index(one(items, "agent_message")) < main.index(tail)
        frames = [parse_frame(m) for m in main[main.index(tail) + 1 :] if isinstance(m, bytes)]
        assert frames and {(t, i) for t, i, _, _ in frames} == {(1, tail["chunk_index"])}
        agent = one(items, "agent_message")["message"]
        assert TAIL not in agent["text"] and agent["heard_text"] is None
        c.send("playback_done", turn_id=1)
        c.until(lambda m: m == {"type": "state", "state": "listening"})
    route = voice.transcript()[-1]["route"]
    assert route["visual"] == "requested" and route["visual_id"] == ready["visual_id"]


def test_a_tail_ready_while_the_answer_is_still_spoken_comes_before_agent_message(voice):
    script = Script(planner=choose("line", "Revenue")).install(voice.fakes.llm)
    script.planner_delay = 0.05
    voice.fakes.tts.delay = 0.25  # speech is slower than the planner
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(SHOW)
        items = c.until("agent_message")
        main = [m for m in items if not (isinstance(m, dict) and m["type"] == "transcript_partial")]
        (tail,) = tail_chunks(items)
        chunks = of(items, "audio_chunk")
        assert chunks[-1] is tail  # after the answer's last chunk
        assert main.index(next(m for m in of(items, "visual") if m["phase"] == "ready")) < main.index(tail)
        assert main.index(tail) < main.index(one(items, "agent_message"))  # the usual order holds


def test_no_tail_when_the_visual_fails(voice):
    Script(planner={"kind": "none", "datasets": ["D1"], "series": []}).install(voice.fakes.llm)
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(SHOW)
        items = c.until(lambda m: isinstance(m, dict) and m.get("type") == "visual" and m["phase"] == "failed")
        items += c.quiet(0.4)
    assert visual_messages(items) == [("visual", "preparing", 1), ("visual", "failed", 1)]
    assert tail_chunks(items) == []  # nothing is promised that didn't come


def test_no_tail_once_the_answer_was_heard(voice):
    script = Script(planner=choose("line", "Revenue")).install(voice.fakes.llm)
    script.planner_delay = 0.6
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(SHOW)
        c.until("agent_message")
        c.send("playback_done", turn_id=1)  # the answer has been heard: the turn is over
        items = c.until(lambda m: isinstance(m, dict) and m.get("type") == "canvas")
        items += c.quiet(0.3)
    assert ("visual", "ready", 1) in visual_messages(items)  # the chart still lands
    assert tail_chunks(items) == []  # but nobody is told about it out of the blue


def test_a_barge_in_while_the_visual_is_planned_stops_the_answer_and_the_new_question_cancels_the_visual(voice):
    script = Script(planner=choose("line", "Revenue")).install(voice.fakes.llm)
    script.planner_delay = 5.0
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(SHOW)
        first = c.until("agent_message")  # the answer's text is complete: its visual is being planned
        assert ("visual", "preparing", 1) in visual_messages(first)
        c.send("barge_in_start", turn_id=1, played_ms=300)
        c.say(NEXT, ms=500)
        items = c.until(lambda m: isinstance(m, dict) and m.get("type") == "agent_message" and m["message"]["seq"] == 4)
    assert one(items, "barge_in") == {"type": "barge_in", "turn_id": 1, "decision": "stop"}
    # the new question needs the model (one request at a time): the visual still being planned gives way
    visual_id = of(first, "visual")[0]["visual_id"]
    cancelled = {"type": "visual", "turn_id": 1, "phase": "failed", "visual_id": visual_id, "detail": "cancelled"}
    assert cancelled in of(items, "visual")
    assert tail_chunks(first + items) == []
    assert voice.fakes.llm.json_cancelled >= 1
    assert of(items, "user_message")[0]["message"]["text"] == "Who is the chairperson of the board?"


def test_a_barge_in_before_the_answer_is_complete_never_starts_a_visual(voice):
    Script(planner=choose("line", "Revenue")).install(voice.fakes.llm)
    voice.fakes.llm.hold_after = 3  # the model is still writing the answer
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(SHOW)
        c.until(lambda m: isinstance(m, dict) and m.get("type") == "delta")
        voice.fakes.llm.hold_after = None
        c.send("stop")
        items = c.until("agent_message")
        items += c.quiet(0.3)
    assert visual_messages(items) == [] and planner_calls(voice.fakes.llm) == []
    assert one(items, "agent_message")["message"]["route"]["stopped"] is True


def test_the_stop_button_cancels_a_visual_being_planned(voice):
    script = Script(planner=choose("line", "Revenue")).install(voice.fakes.llm)
    script.planner_delay = 5.0
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(SHOW)
        c.until("agent_message")
        c.send("stop")
        items = c.until(lambda m: isinstance(m, dict) and m.get("type") == "visual" and m["phase"] == "failed")
    assert of(items, "visual")[-1]["detail"] == "cancelled" and of(items, "visual")[-1]["turn_id"] == 1


def test_spoken_canvas_edits_in_english_and_hindi(voice):
    Script().install(voice.fakes.llm)
    line = line_of_revenue(voice.api, voice.chat, {f"p{d['page']}": d for d in datasets(voice)})
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start("en")
        c.say(EDIT)
        items = c.until("agent_message")
        assert [k for k in kinds(items) if k in ("sources", "visual", "canvas", "delta", "agent_message")] == [
            "sources", "visual", "canvas", "delta", "agent_message",
        ]  # fmt: skip
        assert one(items, "visual")["turn_id"] == 1 and one(items, "visual")["visual"]["kind"] == "bar"
        assert one(items, "visual")["visual_id"] == line["id"]
        assert [ch["text"] for ch in of(items, "audio_chunk")] == ["Done."]
        c.send("playback_done", turn_id=1)
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start("hi")
        c.say(HINDI_EDIT)
        items = c.until("agent_message")
        assert one(items, "agent_message")["message"]["text"] == "हो गया।"
        assert one(items, "canvas")["panels"][0]["kind"] == "table"
        assert [(ch["text"]) for ch in of(items, "audio_chunk")] == ["हो गया।"]
    assert voice.fakes.llm.calls == []  # edits need no model
    assert ("हो गया।", "hi") in voice.fakes.tts.calls


def datasets(voice: Voice) -> list[dict[str, Any]]:
    project_id = voice.api.get(f"/api/chats/{voice.chat}").json()["project_id"]
    return voice.api.get(f"/api/projects/{project_id}/datasets").json()["items"]
