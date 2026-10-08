# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "FlagEmbedding>=1.3", "sentence-transformers>=3.3", "mlx-whisper>=0.4",
#   "kokoro>=0.9.4", "misaki[en]>=0.9.4", "transformers>=4.45",
#   "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#   "qdrant-client>=1.12", "httpx>=0.27", "torch", "numpy", "soundfile", "scipy", "psutil",
# ]
# ///
"""Smoke tests 11 + 13 — end-to-end voice turn latency with every conversation model loaded.

Loads together: mlx-whisper small, BGE-M3, bge-reranker-v2-m3, Kokoro, and
qwen3:4b-instruct in Ollama. Indexes the finance corpus in Qdrant, then for
each spoken question measures:

  STT → [router ∥ speculative retrieval] → rerank (English query) → answer
  first clause → TTS first audio

and reports system memory / swap before and after (smoke test 13).
The VAD end-of-turn wait (~0.65–1.05 s, test 08) comes before this and is
reported separately.

Run:  uv run scripts/smoke/11_end_to_end.py
"""

import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ["HF_HOME"] = str(ROOT / "data/models/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

AUDIO = ROOT / "data/smoke/audio"
FIX = json.loads((Path(__file__).parent / "fixtures/finance_corpus.json").read_text())
LLM = "qwen3:4b-instruct"
OLLAMA = httpx.Client(base_url="http://127.0.0.1:11434", timeout=120)
CLAUSE_END = re.compile(r"[,.!?;:।]\s")
VAD_END_OF_TURN_S = (0.65, 1.05)  # measured in smoke test 08


def local_snapshot(repo_id: str) -> str:
    snaps = Path(os.environ["HF_HOME"]) / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    return str(sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime)[-1])


def sysmem() -> dict:
    vm = psutil.virtual_memory()
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    used_swap = float(re.search(r"used = ([\d.]+)M", swap).group(1)) / 1024
    pressure = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout.strip().splitlines()
    return {"used_gb": (vm.total - vm.available) / 1e9, "available_gb": vm.available / 1e9, "swap_used_gb": used_swap,
            "pressure": pressure[-1] if pressure else "?"}


def pcm16k(name: str) -> np.ndarray:
    a, sr = sf.read(AUDIO / name, dtype="float32")
    if sr != 16_000:
        g = np.gcd(sr, 16_000)
        a = resample_poly(a, 16_000 // g, sr // g).astype(np.float32)
    return a


ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["document_qa", "general_qa", "mixed", "conversation", "resume_document",
                                               "correction", "stop", "backchannel", "clarification"]},
        "retrieve": {"type": "boolean"},
        "lang": {"type": "string", "enum": ["en", "hi"]},
        "query_en": {"type": ["string", "null"]},
    },
    "required": ["intent", "retrieve", "lang", "query_en"],
}
ROUTER_SYSTEM = """Route the latest user utterance for a voice assistant over the user's uploaded documents (an annual report and a contract).
The utterance comes from speech recognition and may contain spelling errors. Output only JSON.
intent: document_qa (facts in the documents) | general_qa | mixed | conversation | resume_document | correction (revises the previous question) | stop | backchannel | clarification.
retrieve: true if document evidence is needed (document_qa, mixed, resume_document, corrections of document questions).
lang: "hi" if the user spoke Hindi, else "en".
query_en: a standalone English search query when retrieve is true (resolve references using the conversation), else null."""

ANSWER_SYSTEM = """You are a friendly voice assistant answering questions about the user's documents.
Reply in 1-2 short spoken sentences, in {lang_name}. No markdown, no lists.
Use only the sources below for document facts; cite nothing aloud. If the sources don't contain the answer, say the documents don't cover it.
Never claim you cannot access the user's documents.

Sources:
{sources}"""


def main() -> int:
    mem0 = sysmem()
    print(f"\n[0] System memory before loading: used {mem0['used_gb']:.1f} GB, available {mem0['available_gb']:.1f} GB, "
          f"swap used {mem0['swap_used_gb']:.2f} GB · {mem0['pressure']}")

    print("\n[1] Load every conversation model")
    t_all = time.perf_counter()
    import mlx_whisper
    from FlagEmbedding import BGEM3FlagModel
    from kokoro import KModel, KPipeline
    from qdrant_client import QdrantClient, models
    from sentence_transformers import CrossEncoder

    t = time.perf_counter()
    stt_path = local_snapshot("mlx-community/whisper-small-mlx")
    stt = lambda pcm: mlx_whisper.transcribe(pcm, path_or_hf_repo=stt_path, fp16=True, verbose=None, condition_on_previous_text=False)  # noqa: E731
    stt(np.zeros(16_000, dtype=np.float32))
    print(f"    mlx-whisper small      {time.perf_counter() - t:5.1f}s")
    t = time.perf_counter()
    emb = BGEM3FlagModel(local_snapshot("BAAI/bge-m3"), use_fp16=True, devices="mps")
    enc = lambda xs: emb.encode(xs, batch_size=16, max_length=1024, return_dense=True, return_sparse=True, return_colbert_vecs=False)  # noqa: E731
    enc(["warm"])
    print(f"    BGE-M3                 {time.perf_counter() - t:5.1f}s")
    t = time.perf_counter()
    rr = CrossEncoder(local_snapshot("BAAI/bge-reranker-v2-m3"), device="mps", max_length=512, model_kwargs={"torch_dtype": torch.float16})
    rr.predict([("warm", "warm")])
    print(f"    reranker               {time.perf_counter() - t:5.1f}s")
    t = time.perf_counter()
    snap = Path(local_snapshot("hexgrad/Kokoro-82M"))
    tts_pipes = {}
    tts_devices = os.environ.get("KOKORO_DEVICES", "cpu").split(",")  # loading both devices costs memory; compare only on demand
    for dev in tts_devices:
        km = KModel(repo_id="hexgrad/Kokoro-82M", config=str(snap / "config.json"), model=str(snap / "kokoro-v1_0.pth")).to(dev).eval()
        tts_pipes[dev] = {"en": KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=km),
                          "hi": KPipeline(lang_code="h", repo_id="hexgrad/Kokoro-82M", model=km)}
    voices = {"en": str(snap / "voices/af_heart.pt"), "hi": str(snap / "voices/hf_alpha.pt")}

    def tts_first(text: str, lang: str, dev: str) -> float:
        t0 = time.perf_counter()
        for _ in tts_pipes[dev][lang](text, voice=voices[lang]):
            return time.perf_counter() - t0
        return time.perf_counter() - t0
    for dev in tts_devices:
        tts_first("Warm up.", "en", dev); tts_first("ठीक है।", "hi", dev)
    print(f"    Kokoro ({'+'.join(tts_devices)})        {time.perf_counter() - t:5.1f}s")
    t = time.perf_counter()
    OLLAMA.post("/api/chat", json={"model": LLM, "messages": [{"role": "user", "content": "hi"}], "stream": False,
                                   "think": False, "keep_alive": "30m", "options": {"num_ctx": 8192, "num_predict": 1}}).raise_for_status()
    print(f"    {LLM:<22} {time.perf_counter() - t:5.1f}s")
    print(f"    total                  {time.perf_counter() - t_all:5.1f}s")

    passages = FIX["passages"]
    P = enc([p["text"] for p in passages])
    qc = QdrantClient(url="http://127.0.0.1:6333")
    coll = "smoke_e2e"
    if qc.collection_exists(coll):
        qc.delete_collection(coll)
    qc.create_collection(coll, vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
                         sparse_vectors_config={"sparse": models.SparseVectorParams()})
    sv = lambda w: models.SparseVector(indices=[int(k) for k in w], values=[float(v) for v in w.values()])  # noqa: E731
    qc.upsert(coll, wait=True, points=[models.PointStruct(id=i, vector={"dense": P["dense_vecs"][i].tolist(), "sparse": sv(P["lexical_weights"][i])},
                                                          payload={"id": p["id"], "page": p["page"], "text": p["text"]}) for i, p in enumerate(passages)])

    def retrieve(q: str) -> list[dict]:
        Q = enc([q])
        return [p.payload for p in qc.query_points(coll, prefetch=[
            models.Prefetch(query=Q["dense_vecs"][0].tolist(), using="dense", limit=10),
            models.Prefetch(query=sv(Q["lexical_weights"][0]), using="sparse", limit=10)],
            query=models.FusionQuery(fusion=models.Fusion.RRF), limit=10, with_payload=True).points]

    mem1 = sysmem()
    ps = next(m for m in OLLAMA.get("/api/ps").json()["models"] if m["name"] == LLM)
    print(f"\n[13] Memory with everything loaded: used {mem1['used_gb']:.1f} GB (+{mem1['used_gb'] - mem0['used_gb']:.1f}), "
          f"available {mem1['available_gb']:.1f} GB, swap {mem1['swap_used_gb']:.2f} GB · {mem1['pressure']}")
    print(f"     torch MPS driver {torch.mps.driver_allocated_memory() / 1e9:.2f} GB · Ollama {ps['size'] / 1e9:.2f} GB "
          f"({ps['size_vram'] / max(ps['size'], 1):.0%} GPU) · this process RSS {psutil.Process().memory_info().rss / 1e9:.2f} GB")

    turns = [
        ("say_en_0.wav", []),
        ("say_en_1.wav", [("user", "What was the company's revenue in fiscal year 2024?"),
                          ("assistant", "[interrupted] FY24 revenue from operations was 4,210 crore, up")]),
        ("say_hi_0.wav", []),
        ("kokoro_hi_4.wav", []),
        ("kokoro_en_3.wav", [("user", "What is the capital of Japan?"), ("assistant", "Tokyo.")]),
    ]
    print("\n[11] Voice turns (timings after end-of-turn is detected)")
    rows = []
    for clip, history in turns:
        pcm = pcm16k(clip)
        r = {"clip": clip}
        t0 = time.perf_counter()
        out = stt(pcm)
        text, lang = out["text"].strip(), out["language"]
        r["stt"] = time.perf_counter() - t0

        spec: dict = {}

        def speculative():
            s = time.perf_counter(); spec["hits"] = retrieve(text); spec["t"] = time.perf_counter() - s
        th = threading.Thread(target=speculative); th.start()

        msgs = [{"role": "system", "content": ROUTER_SYSTEM}]
        if history:
            msgs.append({"role": "user", "content": "Conversation so far:\n" + "\n".join(f"{a}: {b}" for a, b in history)})
        msgs.append({"role": "user", "content": f"Latest utterance: {text}"})
        t1 = time.perf_counter()
        route = json.loads(OLLAMA.post("/api/chat", json={"model": LLM, "messages": msgs, "stream": False, "think": False,
                                                           "format": ROUTE_SCHEMA, "options": {"temperature": 0, "num_ctx": 8192}}).json()["message"]["content"])
        r["router"] = time.perf_counter() - t1
        th.join()
        r["spec_retrieval"] = spec["t"]

        sources, top = "", None
        t2 = time.perf_counter()
        if route["retrieve"]:
            hits = spec["hits"]
            if route.get("query_en") and route["query_en"].strip().lower() != text.strip().lower():
                hits = retrieve(route["query_en"])  # re-retrieve when the rewrite differs (corrections, Hindi)
            scores = rr.predict([(route.get("query_en") or text, h["text"]) for h in hits])
            order = np.argsort(-np.asarray(scores))[:5]
            top = hits[order[0]]["id"]
            sources = "\n".join(f"[S{i + 1}] (p.{hits[j]['page']}) {hits[j]['text']}" for i, j in enumerate(order))
        r["rerank"] = time.perf_counter() - t2

        t3 = time.perf_counter()
        ans, first_clause, t_clause = "", None, None
        with OLLAMA.stream("POST", "/api/chat", json={
                "model": LLM, "stream": True, "think": False, "options": {"temperature": 0.3, "num_ctx": 8192},
                "messages": [{"role": "system", "content": ANSWER_SYSTEM.format(lang_name="Hindi" if route["lang"] == "hi" else "English", sources=sources or "(none: answer from general knowledge)")},
                             *[{"role": "user" if a == "user" else "assistant", "content": b} for a, b in history],
                             {"role": "user", "content": text}]}) as s:
            for line in s.iter_lines():
                if not line:
                    continue
                ch = json.loads(line)
                ans += ch.get("message", {}).get("content", "")
                if first_clause is None:
                    # first speakable chunk: a clause of ≥2 words, or 8 words if no clause boundary arrives
                    m = CLAUSE_END.search(ans + " ")
                    words = ans.split()
                    if m and len(ans[:m.end()].split()) >= 2:
                        first_clause, t_clause = ans[:m.end()].strip(), time.perf_counter() - t3
                    elif len(words) > 8:
                        first_clause, t_clause = " ".join(words[:8]), time.perf_counter() - t3
                if ch.get("done"):
                    break
        r["answer_first_clause"] = t_clause if t_clause is not None else time.perf_counter() - t3
        first_clause = first_clause or ans
        tts_lang = "hi" if route["lang"] == "hi" else "en"
        for dev in tts_devices:
            r[f"tts_{dev}"] = tts_first(first_clause, tts_lang, dev)
        r["tts"] = min(r[f"tts_{dev}"] for dev in tts_devices)
        r["total"] = r["stt"] + r["router"] + r["rerank"] + r["answer_first_clause"] + r["tts"]
        rows.append(r)
        print(f"\n  {clip}: heard {text!r} [{lang}]")
        print(f"    route {route['intent']}/{'retrieve' if route['retrieve'] else 'no-retrieve'}/{route['lang']} q_en={route.get('query_en')!r}  top={top}")
        print(f"    answer: {ans.strip()[:140]!r}")
        print(f"    STT {r['stt']*1000:4.0f} | router {r['router']*1000:4.0f} (∥ retrieval {r['spec_retrieval']*1000:3.0f}) | rerank {r['rerank']*1000:4.0f} | "
              f"first chunk {r['answer_first_clause']*1000:4.0f} | TTS {r['tts']*1000:3.0f} {first_clause!r} "
              f"→ {r['total']:.2f}s")

    qc.delete_collection(coll)
    mem2 = sysmem()
    med = lambda k: statistics.median(x[k] for x in rows)  # noqa: E731
    print("\n=== summary (median over turns) ===")
    for k in ("stt", "router", "rerank", "answer_first_clause", *[f"tts_{d}" for d in tts_devices], "total"):
        print(f"  {k:<20} {med(k) * 1000:6.0f} ms")
    lo, hi = med("total") + VAD_END_OF_TURN_S[0], med("total") + VAD_END_OF_TURN_S[1]
    print(f"\n  user stops talking → first agent audio ≈ {lo:.1f}–{hi:.1f} s (incl. VAD end-of-turn {VAD_END_OF_TURN_S[0]}–{VAD_END_OF_TURN_S[1]} s)")
    print(f"  memory after turns: used {mem2['used_gb']:.1f} GB, available {mem2['available_gb']:.1f} GB, swap {mem2['swap_used_gb']:.2f} GB "
          f"(Δ {mem2['swap_used_gb'] - mem0['swap_used_gb']:+.2f}) · {mem2['pressure']}")
    swap_grew = mem2["swap_used_gb"] - mem0["swap_used_gb"] > 0.5
    print(f"\n  {'FAIL' if swap_grew else 'PASS'}  no significant swapping with all models loaded")
    return 1 if swap_grew else 0


if __name__ == "__main__":
    sys.exit(main())
