"""A voice session transcribes with its chat's vocabulary (DESIGN §9.3): the prompts built from the chat's documents,
and a near-missed name spelled as the documents do before the user message is saved."""

from __future__ import annotations

from app.services.prompts import ACK_TEXTS

from .test_chat_api import REPORT, drain, new_chat
from .test_voice_api import QUESTION, VoiceClient, one, voice  # noqa: F401  (voice is a fixture)

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


def test_a_low_confidence_transcript_is_asked_again(voice):  # noqa: F811
    """Whisper's own confidence on the transcript (avg_logprob far below -1: it decoded words it doesn't believe)
    makes the turn ask the user to say it again instead of answering words that were never said."""
    voice.fakes.stt.confidence[QUESTION] = {"avg_logprob": -1.6, "no_speech_prob": 0.3, "compression_ratio": 1.2}
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start(language="en")
        c.say(QUESTION)
        got = c.until("agent_message")
        c.send("end")
    agent = one(got, "agent_message")["message"]
    assert agent["text"] == ACK_TEXTS["repeat"]["en"] and agent["route"]["intent"] == "clarification"
    assert voice.fakes.llm.calls == []  # nothing was answered
