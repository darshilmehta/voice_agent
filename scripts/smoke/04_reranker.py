# /// script
# requires-python = ">=3.12"
# dependencies = ["sentence-transformers>=3.3", "torch", "numpy"]
# ///
"""Smoke test 04 — bge-reranker-v2-m3 (cross-encoder).

  1. loads fully offline from the project model cache, on the Apple GPU (MPS, fp16)
  2. ranks the right passage first for EN / HI / Hinglish / identifier queries
  3. tells FY24 from FY23 (the classic near-miss that embeddings blur)
  4. abstention signal: unanswerable questions get a low top score
  5. latency for reranking 20 candidates (one conversation turn)
  6. GPU memory

Run:  uv run scripts/smoke/04_reranker.py
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ["HF_HOME"] = str(ROOT / "data/models/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

FIX = json.loads((Path(__file__).parent / "fixtures/finance_corpus.json").read_text())
ABSTAIN_THRESHOLD = 0.3
results: list[tuple[str, bool, str]] = []


def local_snapshot(repo_id: str) -> str:
    """Path of the downloaded snapshot. Passing a path (not a repo id) keeps libraries from contacting the Hub."""
    snaps = Path(os.environ["HF_HOME"]) / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    dirs = sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime) if snaps.exists() else []
    if not dirs:
        raise FileNotFoundError(f"{repo_id} not downloaded; run scripts/setup/download_models.sh")
    return str(dirs[-1])


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


def main() -> int:
    print("\n[1] Load (offline, MPS, fp16)")
    t0 = time.perf_counter()
    model = CrossEncoder(local_snapshot("BAAI/bge-reranker-v2-m3"), device="mps", max_length=512,
                         model_kwargs={"torch_dtype": torch.float16})
    load_s = time.perf_counter() - t0
    passages = FIX["passages"]
    texts = [p["text"] for p in passages]
    ids = [p["id"] for p in passages]

    def scores(q: str, cands: list[str]) -> list[float]:
        return [float(s) for s in model.predict([(q, c) for c in cands], batch_size=32)]

    scores("warm up", texts[:2])
    check("loaded offline from project cache", True, f"{load_s:.1f}s on {model.device}")

    print("\n[2] Ranking (all 13 passages per query)")
    top1 = 0
    margins = []
    for case in FIX["queries"]:
        s = scores(case["q"], texts)
        order = sorted(range(len(s)), key=lambda i: -s[i])
        best, correct = ids[order[0]], ids.index(case["expect"])
        runner = max(s[i] for i in range(len(s)) if i != correct)
        top1 += best == case["expect"]
        margins.append(s[correct] - runner)
        mark = "✓" if best == case["expect"] else "✗"
        print(f"    {mark} {case['q'][:44]!r:<48} correct {s[correct]:.3f}  next best {runner:.3f}  [{case['kind']}]")
    n = len(FIX["queries"])
    check("top-1 accuracy", top1 >= n - 1, f"{top1}/{n}")

    print("\n[3] FY24 vs FY23 discrimination")
    q = "What was the EBITDA margin in FY24?"
    fy24, fy23, defn, generic = (texts[ids.index(i)] for i in ("p01", "p02", "p03", "p08"))
    s24, s23, sdef, sgen = scores(q, [fy24, fy23, defn, generic])
    print(f"        FY24 figure {s24:.3f} | FY23 figure {s23:.3f} | definition {sdef:.3f} | generic margins {sgen:.3f}")
    check("FY24 passage clearly above FY23", s24 > s23 + 0.1, f"gap {s24 - s23:+.3f}")

    print(f"\n[4] Abstention signal (top score < {ABSTAIN_THRESHOLD} → 'not in the documents')")
    answerable_tops = [max(scores(c["q"], texts)) for c in FIX["queries"]]
    for q in FIX["no_answer_queries"]:
        top = max(scores(q, texts))
        check(f"no-answer {q[:36]!r}", top < ABSTAIN_THRESHOLD, f"top score {top:.3f}")
    check(f"answerable top scores [info: not calibrated across languages, abstention uses gap + EN query, §9.2]", True,
          f"lowest top score {min(answerable_tops):.3f}")

    print("\n[5] Latency: rerank 20 candidates (one turn)")
    cands = (texts * 2)[:20]
    lat = []
    for _ in range(15):
        t0 = time.perf_counter(); scores(q, cands); lat.append((time.perf_counter() - t0) * 1000)
    check("rerank 20 p50 < 300 ms", statistics.median(lat) < 300, f"p50 {statistics.median(lat):.0f} ms, max {max(lat):.0f} ms")

    print("\n[6] Memory")
    check("MPS memory", True, f"allocated {torch.mps.current_allocated_memory() / 1e9:.2f} GB, "
                              f"driver {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
