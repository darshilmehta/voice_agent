"""A voice session transcribes with its chat's vocabulary (DESIGN §9.3): the prompts built from the chat's documents,
and a near-missed name spelled as the documents do before the user message is saved."""

from __future__ import annotations

from .test_chat_api import REPORT, drain, new_chat
from .test_voice_api import QUESTION, VoiceClient, one

HEARD = "What was Zefyrion's revenue in FY24?"
MEANT = "What was Zephyrion's revenue in FY24?"


def test_the_utterance_is_transcribed_with_the_chats_names(make_app, fakes):
    fakes.llm.reply = "Revenue was Rs 4,210 crore [S1]."
    fakes.stt.scripts[QUESTION] = HEARD
    with make_app() as api:
        project = api.post("/api/projects", json={"name": "Zephyrion"}).json()["id"]
        r = api.post(
            f"/api/projects/{project}/documents",
            files={"file": ("zephyrion_annual_report.txt", REPORT.encode(), "text/plain")},
        )
        assert r.status_code == 202, r.text
        drain(api)
        chat = new_chat(api, project)
        with api.websocket_connect(f"/ws/chats/{chat}/voice") as ws:
            c = VoiceClient(ws)
            c.start(language="en")
            c.quiet(0.3)  # the chat's vocabulary is built in the background
            c.say(QUESTION)
            got = c.until("agent_message")
            c.send("end")
    assert one(got, "user_message")["message"]["text"] == MEANT
    prompted = [call["prompts"] for call in fakes.stt.calls if call["prompts"]]
    assert prompted and all("Zephyrion" in p["en"] for p in prompted)
    assert prompted[0]["hi"].startswith("Zephyrion")  # no Devanagari spelling: the fake LLM gives none
