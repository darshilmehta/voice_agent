"""The router in a voice session (fake speech): an acknowledgement said while the agent is idle gets a two-word reply
(never an abstention), a second one gets nothing, a spoken "stop" says nothing, and a Hinglish question is answered
in Hindi through the English search query."""

from __future__ import annotations

from .test_conversation_turns import routes
from .test_voice_api import QUESTION, VoiceClient, of, one, voice  # noqa: F401  (voice is a fixture)

OKAY, THEEK_HAI, STOP, HINGLISH = 70, 71, 72, 73


def test_acknowledgements_while_idle_get_a_short_reply_or_nothing(voice):  # noqa: F811
    voice.fakes.stt.scripts.update({OKAY: "Okay.", THEEK_HAI: "theek hai", STOP: "Stop."})
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start()
        c.say(QUESTION)
        c.until("agent_message")
        c.send("playback_done", turn_id=1)

        c.say(OKAY)
        items = c.until("agent_message")
        agent = one(items, "agent_message")["message"]
        assert (agent["text"], agent["route"]["intent"], agent["route"]["abstained"]) == (
            "Anything else?",
            "backchannel",
            False,
        )
        assert [ch["text"] for ch in of(items, "audio_chunk")] == ["Anything else?"]  # spoken
        c.send("playback_done", turn_id=2)

        c.say(THEEK_HAI)  # a second acknowledgement right after "Anything else?": nothing is said
        items = c.until("agent_message")
        notice = one(items, "agent_message")["message"]
        assert (notice["role"], notice["route"]["answer"]) == ("event", "silent")
        assert of(items, "audio_chunk") == [] and of(items, "delta") == []

        c.say(STOP)
        items = c.until("agent_message")
        notice = one(items, "agent_message")["message"]
        assert (notice["role"], notice["text"], notice["route"]["intent"]) == ("event", "Stopped", "stop")
        assert of(items, "audio_chunk") == []
    assert all(m["route"] is None or m["route"]["abstained"] is False for m in voice.transcript()[2:])


def test_a_hinglish_question_is_answered_in_hindi_with_citations(voice):  # noqa: F811
    voice.fakes.stt.scripts[HINGLISH] = "FY24 mein EBITDA margin kitna tha?"
    voice.fakes.reranker.scorer = lambda q, passage: 0.9 if "18.2%" in passage and "EBITDA" in q else 0.01
    voice.fakes.llm.route = routes(
        {
            "FY24 mein EBITDA margin kitna tha?": {
                "intent": "document_qa",
                "query": "What was the EBITDA margin in FY24?",
            }
        }
    )
    voice.fakes.llm.reply = "वित्त वर्ष 2024 में EBITDA मार्जिन 18.2% था [S1]।"
    with voice.connect() as ws:
        c = VoiceClient(ws)
        c.start(language=None)
        c.say(HINGLISH)
        items = c.until("agent_message")
    agent = one(items, "agent_message")["message"]
    assert agent["language"] == "hi" and agent["citations"][0]["source_id"] == "S1"
    assert agent["route"]["query_en"] == "What was the EBITDA margin in FY24?"
    assert voice.fakes.reranker.calls[-1][0] == "What was the EBITDA margin in FY24?"
    assert {lang for _, lang in voice.fakes.tts.calls} == {"hi"}
