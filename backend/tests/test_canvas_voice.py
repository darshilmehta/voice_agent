"""The canvas in a voice session (docs/DESIGN.md §12.1, §3.10), fake speech: the answer's visual arrives as `visual` and
`canvas` messages with the turn's id. Its draft is built as soon as the retrieval returns and held until the answer's
first audio, then sent at once (never before it); a draft the planner refines is replaced in place, possibly after
`agent_message`. The spoken tail ("It's on screen now.") only for a visual that stays and while the answer is still
being heard; barge-in, a new question or the stop button during the refinement keep the draft; a turn cut before its
first audio never shows one. Spoken canvas edits (English and Hindi)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest

from app.services.voice.fillers import VISUAL_TAILS
from app.services.voice.protocol import parse_frame

from .canvas_turn_helpers import Script, choose, choose_table, line_of_revenue, planner_calls, report_chat
from .test_voice_api import Item, Voice, VoiceClient, kinds, of, one

SHOW, NEXT, EDIT, HINDI_EDIT, UNSURE = 30, 40, 50, 60, 70  # tones: what the user "says"
REPLY = "Revenue rose in every quarter of FY24 [S1]. It ended the year at its highest level."
TAIL = VISUAL_TAILS["en"]


@pytest.fixture
def voice(make_app, fakes) -> Iterator[Voice]:
    fakes.llm.reply = REPLY
    fakes.stt.scripts.update(
        {
            SHOW: "Show me revenue by quarter",  # a confident draft: the line of quarterly revenue
            NEXT: "Who is the chairperson of the board?",
            EDIT: "Make it a bar chart.",
            HINDI_EDIT: "इसे टेबल में दिखाओ",
            UNSURE: "Show me revenue",  # which table? a draft for the planner to check
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


def main_of(items: list[Item]) -> list[Item]:
    return [m for m in items if not (isinstance(m, dict) and m["type"] == "transcript_partial")]


def refined(m: Item) -> bool:
    """The canvas once the planner's line chart replaced the draft."""
    return isinstance(m, dict) and m.get("type") == "canvas" and m["panels"][0]["kind"] == "line"


def canvas_kinds(voice: Voice) -> list[str]:
    return [p["kind"] for p in voice.api.get(f"/api/chats/{voice.chat}/canvas").json()["panels"]]


def test_the_draft_is_sent_with_the_first_audio_and_the_tail_follows_the_answer(voice):
    Script(planner=choose("line", "Revenue")).install(voice.fakes.llm)
    voice.fakes.tts.delay = 0.2  # the first audio takes a while: the draft is ready long before it, and waits
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(SHOW)
        items = c.until("agent_message")
        main = main_of(items)
        chunks = of(items, "audio_chunk")
        first_audio, second_audio = main.index(chunks[0]), main.index(chunks[1])
        preparing = main.index(next(m for m in of(items, "visual") if m["phase"] == "preparing"))
        ready = next(m for m in of(items, "visual") if m["phase"] == "ready")
        # never before the first audio; sent at once with it, not after the answer
        assert first_audio < preparing < main.index(ready) < second_audio
        assert visual_messages(items) == [("visual", "preparing", 1), ("visual", "ready", 1), ("canvas", None, 1)]
        assert ready["visual"]["kind"] == "line"
        # ready while the answer is spoken: the tail is its last chunk, before agent_message
        (tail,) = tail_chunks(items)
        assert chunks[-1] is tail and tail["text"] == TAIL and tail["turn_id"] == 1
        assert main.index(tail) < main.index(one(items, "agent_message"))
        frames = [parse_frame(m) for m in main[main.index(tail) + 1 :] if isinstance(m, bytes)]
        assert frames and {(t, i) for t, i, _, _ in frames} == {(1, tail["chunk_index"])}
        agent = one(items, "agent_message")["message"]
        assert TAIL not in agent["text"] and agent["heard_text"] is None
        c.send("playback_done", turn_id=1)
        c.until(lambda m: m == {"type": "state", "state": "listening"})
    assert planner_calls(voice.fakes.llm) == []  # a confident draft: no model call
    route = voice.transcript()[-1]["route"]
    assert route["visual"] == "requested" and route["visual_id"] == ready["visual_id"]
    assert route["visual_plan"]["draft"] == "confident"


def test_the_same_chart_again_takes_the_old_panels_place_without_a_tail(voice):
    """§12.1 "A visual already on the canvas": nothing new appears, so "It's on screen now." isn't said."""
    Script(planner=choose("line", "Revenue")).install(voice.fakes.llm)
    voice.fakes.tts.delay = 0.2
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(SHOW)
        first = c.until("agent_message")
        c.send("playback_done", turn_id=1)
        c.until(lambda m: m == {"type": "state", "state": "listening"})
        c.say(SHOW)
        items = c.until(lambda m: isinstance(m, dict) and m.get("type") == "agent_message" and m["message"]["seq"] > 2)
        ready = next(m for m in of(items, "visual") if m["phase"] == "ready" and m["turn_id"] == 2)
        first_id = next(m for m in of(first, "visual") if m["phase"] == "ready")["visual_id"]
        assert ready["replaces"] == first_id and ready["reuse"] == "same"
        assert not [m for m in tail_chunks(items) if m["turn_id"] == 2]
    assert canvas_kinds(voice) == ["line"]


def test_a_refined_visual_replaces_the_draft_in_place_after_the_answer(voice):
    script = Script(planner=choose_table("line", "Q1 FY24", "Revenue")).install(voice.fakes.llm)
    script.planner_delay = 0.3
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(UNSURE)
        items = c.until(refined)
        items += c.quiet(0.3)
    readies = [m for m in of(items, "visual") if m["phase"] == "ready"]
    assert [r["visual"]["kind"] for r in readies] == ["donut", "line"]  # the draft, then the planner's chart
    assert readies[0]["visual_id"] == readies[1]["visual_id"]  # in place: one panel for the turn
    main = main_of(items)
    assert main.index(of(items, "audio_chunk")[0]) < main.index(readies[0])
    assert canvas_kinds(voice) == ["line"]
    # the draft wasn't sure: the tail waited for the planner's verdict, and the answer was still playing
    assert [m["text"] for m in tail_chunks(items)] == [TAIL]


def test_no_tail_when_the_planner_withdraws_the_draft(voice):
    Script(planner={"kind": "none", "datasets": ["D1"], "series": []}).install(voice.fakes.llm)
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(UNSURE)
        items = c.until(lambda m: isinstance(m, dict) and m.get("type") == "visual" and m["phase"] == "failed")
        items += c.quiet(0.4)
    assert visual_messages(items) == [
        ("visual", "preparing", 1),
        ("visual", "ready", 1),
        ("canvas", None, 1),
        ("visual", "failed", 1),
        ("canvas", None, 1),
    ]
    assert tail_chunks(items) == []  # nothing is promised that didn't stay
    assert canvas_kinds(voice) == []


def test_an_answer_that_abstains_before_its_first_audio_shows_no_visual(voice):
    Script(planner=choose("line", "Revenue")).install(voice.fakes.llm)
    # the answer itself abstains (B9), on what no table states (a denial of what the chart shows is asked again)
    voice.fakes.llm.reply = "The documents don't cover revenue for FY25."
    voice.fakes.tts.delay = 0.3  # its text is complete before its first audio: the draft is withdrawn while held
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(UNSURE)
        items = c.until("agent_message")
        items += c.quiet(0.4)
    assert visual_messages(items) == [] and tail_chunks(items) == []
    assert canvas_kinds(voice) == [] and planner_calls(voice.fakes.llm) == []


def test_no_tail_once_the_answer_was_heard(voice):
    script = Script(planner=choose_table("line", "Q1 FY24", "Revenue")).install(voice.fakes.llm)
    script.planner_delay = 0.6
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(UNSURE)
        c.until("agent_message")
        c.send("playback_done", turn_id=1)  # the answer has been heard: the turn is over
        items = c.until(refined)
        items += c.quiet(0.3)
    assert ("visual", "ready", 1) in visual_messages(items)  # the refined chart still lands
    assert tail_chunks(items) == []  # but nobody is told about it out of the blue


def test_a_barge_in_during_the_refinement_keeps_the_draft_and_the_new_question_cancels_the_planner(voice):
    script = Script(planner=choose_table("line", "Q1 FY24", "Revenue")).install(voice.fakes.llm)
    script.planner_delay = 5.0
    voice.fakes.llm.delay = 0.03  # the answer's words take a moment, as on the real model: the draft is ready
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(UNSURE)
        first = c.until("agent_message")  # the answer's text is complete: the planner is refining the draft
        assert visual_messages(first)[:2] == [("visual", "preparing", 1), ("visual", "ready", 1)]
        c.send("barge_in_start", turn_id=1, played_ms=300)
        c.say(NEXT, ms=500)
        items = c.until(lambda m: isinstance(m, dict) and m.get("type") == "agent_message" and m["message"]["seq"] == 4)
        items += c.quiet(0.2)
    assert one(items, "barge_in") == {"type": "barge_in", "turn_id": 1, "decision": "stop"}
    # the new question needs the model (one request at a time): the refinement gives way, the draft stays
    assert voice.fakes.llm.json_cancelled >= 1
    assert [m for m in of(items, "visual") if m["phase"] == "failed"] == []
    assert canvas_kinds(voice) == ["donut"]
    assert tail_chunks(first + items) == []
    assert of(items, "user_message")[0]["message"]["text"] == "Who is the chairperson of the board?"
    first_answer = voice.transcript()[1]["route"]
    assert first_answer["visual_status"] == "ready" and first_answer["visual_plan"]["planner"] == "cancelled"


def test_a_stop_before_the_first_audio_never_shows_the_draft(voice):
    Script(planner=choose("line", "Revenue")).install(voice.fakes.llm)
    # the model is still writing the first words (the first is out, 3 words behind it: too few for speech)
    voice.fakes.llm.reply = "Revenue rose in every quarter of FY24, from 1,742 to 1,933 in the fourth quarter [S1]."
    voice.fakes.llm.hold_after = 5
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(SHOW)
        c.until(lambda m: isinstance(m, dict) and m.get("type") == "delta")
        voice.fakes.llm.hold_after = None
        c.send("stop")
        items = c.until("agent_message")
        items += c.quiet(0.3)
    assert of(items, "audio_chunk") == []
    assert visual_messages(items) == [] and planner_calls(voice.fakes.llm) == []
    assert one(items, "agent_message")["message"]["route"]["stopped"] is True
    assert canvas_kinds(voice) == []  # the draft built for it is withdrawn: nobody saw it


def test_the_stop_button_during_the_refinement_keeps_the_draft(voice):
    script = Script(planner=choose_table("line", "Q1 FY24", "Revenue")).install(voice.fakes.llm)
    script.planner_delay = 5.0
    voice.fakes.llm.delay = 0.03  # the answer's words take a moment, as on the real model: the draft is ready
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(UNSURE)
        c.until("agent_message")
        deadline = time.monotonic() + 5
        while not script.planner_at and time.monotonic() < deadline:  # the planner is refining the draft
            time.sleep(0.01)
        c.send("stop")
        items = c.quiet(0.5)
    assert voice.fakes.llm.json_cancelled >= 1  # the planner's request is closed
    assert [m for m in of(items, "visual") if m["phase"] == "failed"] == []
    assert canvas_kinds(voice) == ["donut"]


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
