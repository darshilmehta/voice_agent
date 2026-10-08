# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27", "pydantic>=2.7"]
# ///
"""Smoke test 01 — Ollama + Qwen3.

For each model (default: qwen3:4b, qwen3:8b):

  1. server is local (loopback) and the model is a local tag
  2. cold load time
  3. voice-style streaming answer with thinking OFF:
     time to first token, time to first complete sentence, tokens/sec,
     and no reasoning leaked into the output
  4. Hindi question → Hindi (Devanagari) answer
  5. router: JSON-schema constrained output on EN / HI / Hinglish turns,
     including interruption cases (correction, stop, backchannel)
  6. memory footprint and GPU offload (/api/ps)

Run:  uv run scripts/smoke/01_ollama.py [model ...]
"""

import json
import re
import statistics
import sys
import time
from typing import Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

OLLAMA = "http://127.0.0.1:11434"
MODELS = sys.argv[1:] or ["qwen3:4b-instruct", "qwen3:8b"]
NUM_CTX = 8192
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
SENTENCE_END = re.compile(r"[.!?।]\s")
# Reasoning that leaks into the answer text instead of a separate thinking channel.
REASONING_LEAK = re.compile(r"^(okay|ok|alright|hmm)\b|the user (wants|is asking|asked)|let me (think|recall)", re.I)

client = httpx.Client(base_url=OLLAMA, timeout=300)
results: list[tuple[str, str, bool, str]] = []


def check(model: str, name: str, ok: bool, detail: str = "") -> None:
    results.append((model, name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


class TurnRoute(BaseModel):
    intent: Literal[
        "document_qa", "general_qa", "mixed", "conversation", "resume_document",
        "correction", "stop", "backchannel", "clarification",
    ]
    needs_retrieval: bool
    rewritten_query: str | None
    topic: str
    is_topic_shift: bool
    response_language: Literal["en", "hi"]
    confidence: float = Field(ge=0, le=1)


class SlimRoute(BaseModel):
    """Fewer, shorter fields: fewer output tokens → lower router latency."""
    intent: Literal[
        "document_qa", "general_qa", "mixed", "conversation", "resume_document",
        "correction", "stop", "backchannel", "clarification",
    ]
    retrieve: bool
    lang: Literal["en", "hi"]
    query: str | None


ROUTER_SYSTEM = """You are the turn router for a voice assistant that answers questions about the user's uploaded documents.
Classify the user's latest utterance. Output only JSON matching the schema.

intent:
- document_qa: asks about facts in the uploaded documents
- general_qa: general-knowledge question unrelated to the documents
- mixed: combines document facts with general knowledge or opinion
- conversation: chit-chat, jokes, greetings
- resume_document: explicitly returns to the documents after a digression
- correction: revises or replaces the previous / interrupted question
- stop: asks the assistant to stop talking
- backchannel: short acknowledgement while the assistant talks ("mm-hmm", "okay", "haan") — not a new request
- clarification: the request is too ambiguous to act on

needs_retrieval: true when document evidence is needed (document_qa, mixed, resume_document, and corrections of document questions).
rewritten_query: a standalone search query when the utterance depends on context (follow-ups, corrections); otherwise null.
response_language / lang: "hi" if the user spoke Hindi (Devanagari or romanized Hinglish), else "en".
retrieve = needs_retrieval; query = rewritten_query (null unless needed).
"""

# (context_turns, utterance, accepted_intents, expected_needs_retrieval, expected_language)
ROUTER_CASES = [
    ([], "What was the company's revenue in FY24 according to the report?", {"document_qa"}, True, "en"),
    ([], "What's the capital of Japan?", {"general_qa"}, False, "en"),
    ([], "रिपोर्ट के अनुसार FY24 में कंपनी का मुनाफा कितना था?", {"document_qa"}, True, "hi"),
    ([("user", "What's the capital of Japan?"), ("assistant", "Tokyo.")],
     "Okay, back to the report — what was the EBITDA margin?", {"resume_document", "document_qa"}, True, "en"),
    ([("user", "What was revenue in FY24?"), ("assistant", "[interrupted] FY24 revenue was 4.2 billion, up from")],
     "No wait, I meant FY25.", {"correction"}, True, "en"),
    ([("user", "Summarize the risk section."), ("assistant", "[speaking] The report lists three main risks. First,")],
     "mm-hmm", {"backchannel"}, False, "en"),
    ([("user", "Summarize the risk section."), ("assistant", "[speaking] The report lists three main risks. First,")],
     "Stop, stop.", {"stop"}, False, "en"),
    ([], "Tell me a joke.", {"conversation"}, False, "en"),
    ([], "EBITDA ka matlab kya hota hai, simple mein batao", {"general_qa"}, False, "hi"),
    ([], "The report says the EBITDA margin is 18%. Is that considered good for this industry?", {"mixed"}, True, "en"),
]


def chat_stream(model: str, messages: list[dict], **extra) -> dict:
    """Stream a chat; return timings and full text."""
    body = {"model": model, "messages": messages, "stream": True, "think": False,
            "options": {"temperature": 0.3, "num_ctx": NUM_CTX}, **extra}
    t0 = time.perf_counter()
    first_token = first_sentence = None
    text, final = "", {}
    with client.stream("POST", "/api/chat", json=body) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                if first_token is None:
                    first_token = time.perf_counter() - t0
                text += piece
                if first_sentence is None and SENTENCE_END.search(text + " "):
                    first_sentence = time.perf_counter() - t0
            if chunk.get("done"):
                final = chunk
    final.update(text=text, ttft=first_token, first_sentence=first_sentence, wall=time.perf_counter() - t0)
    return final


def route(model: str, context: list[tuple[str, str]], utterance: str, schema: type[BaseModel]) -> tuple[str, float, dict]:
    msgs = [{"role": "system", "content": ROUTER_SYSTEM}]
    if context:
        convo = "\n".join(f"{role}: {text}" for role, text in context)
        msgs.append({"role": "user", "content": f"Conversation so far:\n{convo}"})
    msgs.append({"role": "user", "content": f"Latest utterance: {utterance}"})
    t0 = time.perf_counter()
    r = client.post("/api/chat", json={
        "model": model, "messages": msgs, "stream": False, "think": False,
        "format": schema.model_json_schema(), "options": {"temperature": 0, "num_ctx": NUM_CTX},
    })
    r.raise_for_status()
    d = r.json()
    stats = {"prompt_tokens": d.get("prompt_eval_count", 0), "prompt_ms": d.get("prompt_eval_duration", 0) / 1e6,
             "out_tokens": d.get("eval_count", 0), "gen_ms": d.get("eval_duration", 0) / 1e6}
    return d["message"]["content"], time.perf_counter() - t0, stats


def unload(model: str) -> None:
    client.post("/api/generate", json={"model": model, "keep_alive": 0})
    for _ in range(30):
        if not any(m["name"] == model for m in client.get("/api/ps").json().get("models", [])):
            return
        time.sleep(0.5)


def router_suite(model: str, schema: type[BaseModel]) -> None:
    name = schema.__name__
    print(f"\n[router: {name}]")
    route(model, [], "warm up", schema)
    valid = intent_ok = retr_ok = lang_ok = 0
    latencies, stats = [], []
    for ctx, utt, intents, need, lang in ROUTER_CASES:
        raw, dt, st = route(model, ctx, utt, schema)
        latencies.append(dt); stats.append(st)
        try:
            r = schema.model_validate_json(raw)
        except ValidationError as e:
            print(f"    ✗ invalid JSON for {utt!r}: {e.errors()[0]['msg']}")
            continue
        valid += 1
        got_need = r.needs_retrieval if schema is TurnRoute else r.retrieve
        got_lang = r.response_language if schema is TurnRoute else r.lang
        query = r.rewritten_query if schema is TurnRoute else r.query
        ok_i, ok_r, ok_l = r.intent in intents, got_need == need, got_lang == lang
        intent_ok += ok_i; retr_ok += ok_r; lang_ok += ok_l
        mark = "✓" if ok_i and ok_r and ok_l else "✗"
        print(f"    {mark} {dt * 1000:4.0f} ms (prompt {st['prompt_tokens']:>3} tok {st['prompt_ms']:4.0f} ms | out {st['out_tokens']:>3} tok {st['gen_ms']:4.0f} ms)"
              f"  {r.intent:<15} retr={str(got_need):<5} lang={got_lang}  ← {utt[:40]!r}" + (f" q={query!r}" if query else ""))
    n = len(ROUTER_CASES)
    med = statistics.median
    check(model, f"{name} JSON valid", valid == n, f"{valid}/{n}")
    check(model, f"{name} intent accuracy ≥ 80%", intent_ok / n >= 0.8, f"{intent_ok}/{n}")
    check(model, f"{name} needs_retrieval accuracy ≥ 80%", retr_ok / n >= 0.8, f"{retr_ok}/{n}")
    check(model, f"{name} language accuracy = 100%", lang_ok == n, f"{lang_ok}/{n}")
    # Full schema is known to be too slow (output-length bound, §9.1); only the slim schema is held to the target.
    check(model, f"{name} latency (median){' < 1.5 s' if schema is SlimRoute else ' [info]'}", schema is not SlimRoute or med(latencies) < 1.5,
          f"{med(latencies) * 1000:.0f} ms; median output {med(x['out_tokens'] for x in stats):.0f} tok, "
          f"prompt {med(x['prompt_ms'] for x in stats):.0f} ms, generation {med(x['gen_ms'] for x in stats):.0f} ms")


def test_model(model: str) -> None:
    print(f"\n=== {model} ===")
    check(model, "local model tag", not model.endswith("-cloud") and any(
        m["name"] == model for m in client.get("/api/tags").json()["models"]))

    print("\n[cold load]")
    unload(model)
    t0 = time.perf_counter()
    r = client.post("/api/generate", json={"model": model, "prompt": "", "keep_alive": "10m"})
    r.raise_for_status()
    check(model, "cold load", True, f"{time.perf_counter() - t0:.1f}s")

    print("\n[voice-style answer, thinking off]")
    voice_sys = ("You are a friendly voice assistant. Reply in 1-3 short spoken sentences. "
                 "No markdown, no lists.")
    runs = [chat_stream(model, [{"role": "system", "content": voice_sys},
                                {"role": "user", "content": q}])
            for q in ["Explain what EBITDA means, simply.",
                      "Why do companies report revenue growth year over year?",
                      "What is a good way to read an annual report quickly?"]]
    ttft = statistics.median(x["ttft"] for x in runs)
    fs = statistics.median(x["first_sentence"] or x["wall"] for x in runs)
    tps = statistics.median(x["eval_count"] / (x["eval_duration"] / 1e9) for x in runs)
    leaked = [x["text"][:60] for x in runs if "<think>" in x["text"] or REASONING_LEAK.search(x["text"][:200])]
    check(model, "time to first token (warm, median)", ttft < 1.0, f"{ttft * 1000:.0f} ms")
    check(model, "time to first sentence (median)", fs < 2.0, f"{fs * 1000:.0f} ms")
    check(model, "generation speed (median)", tps > 15, f"{tps:.1f} tok/s")
    check(model, "no reasoning leaked into answer", not leaked, f"{len(leaked)}/{len(runs)} leaked" if leaked else "")
    print(f"        sample: {runs[0]['text'].strip()[:160]!r}")

    print("\n[Hindi]")
    hi = chat_stream(model, [{"role": "system", "content": voice_sys + " Answer in the user's language."},
                             {"role": "user", "content": "EBITDA क्या होता है? आसान भाषा में समझाइए।"}])
    ratio = len(DEVANAGARI.findall(hi["text"])) / max(1, len(re.sub(r"\s", "", hi["text"])))
    hi_tps = hi["eval_count"] / (hi["eval_duration"] / 1e9)
    print(f"        Hindi: {hi['eval_count']} tokens for {len(hi['text'])} chars ({hi['eval_count'] / max(1, len(hi['text'])):.2f} tok/char), {hi_tps:.1f} tok/s")
    check(model, "Hindi question → Devanagari answer", ratio > 0.5, f"{ratio:.0%} Devanagari, first sentence {hi['first_sentence'] or hi['wall']:.2f}s")
    print(f"        sample: {hi['text'].strip()[:160]!r}")

    for schema in (TurnRoute, SlimRoute):
        router_suite(model, schema)

    print("\n[memory]")
    ps = next(m for m in client.get("/api/ps").json()["models"] if m["name"] == model)
    gpu = ps["size_vram"] / ps["size"] if ps["size"] else 0
    check(model, "fully on GPU", gpu > 0.99, f"{ps['size'] / 1e9:.2f} GB loaded, {gpu:.0%} on GPU, ctx {NUM_CTX}")
    unload(model)


def main() -> int:
    v = client.get("/api/version").json()["version"]
    print(f"Ollama {v} at {OLLAMA}")
    for m in MODELS:
        test_model(m)

    print("\n=== summary ===")
    for m in MODELS:
        rows = [r for r in results if r[0] == m]
        print(f"  {m}: {sum(r[2] for r in rows)}/{len(rows)} checks passed")
    return 0 if all(r[2] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
