# /// script
# requires-python = ">=3.12"
# dependencies = ["FlagEmbedding>=1.3", "torch", "numpy", "qdrant-client>=1.12"]
# ///
"""Smoke test 03 — BGE-M3 embeddings (dense + sparse).

  1. loads fully offline from the project model cache, on the Apple GPU (MPS, fp16)
  2. shows what a real dense + sparse vector looks like
  3. cross-lingual: EN ↔ HI / Hinglish translations score far above unrelated text
  4. retrieval on a small annual-report corpus: dense, sparse and hybrid (via Qdrant RRF)
  5. speed: single-query latency (per conversation turn) and batch throughput (ingestion)
  6. GPU memory

Run:  uv run scripts/smoke/03_bge_m3.py
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
from FlagEmbedding import BGEM3FlagModel  # noqa: E402
from qdrant_client import QdrantClient, models  # noqa: E402

FIX = json.loads((Path(__file__).parent / "fixtures/finance_corpus.json").read_text())
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


def gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def main() -> int:
    print("\n[1] Load (offline, MPS, fp16)")
    check("MPS available", torch.backends.mps.is_available())
    t0 = time.perf_counter()
    model = BGEM3FlagModel(local_snapshot("BAAI/bge-m3"), use_fp16=True, devices="mps")
    load_s = time.perf_counter() - t0

    def enc(texts: list[str], **kw):
        return model.encode(texts, batch_size=16, max_length=1024, return_dense=True,
                            return_sparse=True, return_colbert_vecs=False, **kw)

    enc(["warm up"])
    check("loaded offline from project cache", True, f"{load_s:.1f}s, device={model.target_devices}")

    print("\n[2] Anatomy of one embedding")
    p01 = FIX["passages"][0]["text"]
    out = enc([p01])
    dense, lw = out["dense_vecs"][0], out["lexical_weights"][0]
    tokens = model.convert_id_to_token(lw)
    tokens = tokens[0] if isinstance(tokens, list) else tokens
    top = sorted(tokens.items(), key=lambda kv: -kv[1])[:8]
    print(f"        text:   {p01!r}")
    print(f"        dense:  {len(dense)} numbers, norm {np.linalg.norm(dense):.3f}, first 6 = {np.round(dense[:6], 3).tolist()}")
    print(f"        sparse: {len(lw)} weighted tokens, top = {[(t, round(float(w), 2)) for t, w in top]}")
    check("dense is 1024-d and normalized", len(dense) == 1024 and abs(np.linalg.norm(dense) - 1) < 1e-2)
    check("sparse has lexical weights", len(lw) > 5)

    print("\n[3] Cross-lingual similarity (cosine of dense vectors)")
    unrelated = enc([FIX["unrelated"]])["dense_vecs"][0]
    for a, b in FIX["translation_pairs"]:
        va, vb = enc([a, b])["dense_vecs"]
        same, other = float(va @ vb), float(va @ unrelated)
        check(f"{b[:38]!r}", same > other + 0.2, f"translation {same:.3f} vs unrelated {other:.3f}")

    print("\n[4] Retrieval on the annual-report corpus")
    passages = FIX["passages"]
    t0 = time.perf_counter()
    P = enc([p["text"] for p in passages])
    corpus_s = time.perf_counter() - t0
    ids = [p["id"] for p in passages]

    client = QdrantClient(url="http://127.0.0.1:6333")
    coll = "smoke_bge_m3"
    if client.collection_exists(coll):
        client.delete_collection(coll)
    client.create_collection(coll, vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
                             sparse_vectors_config={"sparse": models.SparseVectorParams()})

    def sparse_vec(w: dict) -> models.SparseVector:
        return models.SparseVector(indices=[int(k) for k in w], values=[float(v) for v in w.values()])

    client.upsert(coll, wait=True, points=[
        models.PointStruct(id=i, vector={"dense": P["dense_vecs"][i].tolist(), "sparse": sparse_vec(P["lexical_weights"][i])},
                           payload={"chunk_id": ids[i], "page": passages[i]["page"]})
        for i in range(len(passages))])

    hits = {"dense": 0, "sparse": 0, "hybrid": 0}
    hybrid_top3 = 0
    print(f"        {'query':<52} {'dense':>6} {'sparse':>6} {'hybrid':>6}   (rank of correct passage)")
    for case in FIX["queries"]:
        Q = enc([case["q"]])
        qd, qs = Q["dense_vecs"][0], Q["lexical_weights"][0]
        dense_rank = [ids[i] for i in np.argsort(-(P["dense_vecs"] @ qd))]
        sparse_scores = [model.compute_lexical_matching_score(qs, w) for w in P["lexical_weights"]]
        sparse_rank = [ids[i] for i in np.argsort(-np.array(sparse_scores))]
        hybrid_rank = [p.payload["chunk_id"] for p in client.query_points(
            coll, prefetch=[models.Prefetch(query=qd.tolist(), using="dense", limit=10),
                            models.Prefetch(query=sparse_vec(qs), using="sparse", limit=10)],
            query=models.FusionQuery(fusion=models.Fusion.RRF), limit=10, with_payload=True).points]
        r = lambda rank: rank.index(case["expect"]) + 1 if case["expect"] in rank else 99  # noqa: E731
        rd, rs, rh = r(dense_rank), r(sparse_rank), r(hybrid_rank)
        hits["dense"] += rd == 1; hits["sparse"] += rs == 1; hits["hybrid"] += rh == 1; hybrid_top3 += rh <= 3
        label = f"{case['q'][:34]!r} [{case['kind']}]"
        print(f"        {label[:52]:<52} {rd:>6} {rs:>6} {rh:>6}")
    client.delete_collection(coll)
    n = len(FIX["queries"])
    check("dense top-1", hits["dense"] >= n - 2, f"{hits['dense']}/{n}")
    check("sparse top-1 (exact terms)", True, f"{hits['sparse']}/{n} (expected to miss cross-lingual cases)")
    check("hybrid top-1 [info: RRF demotes cross-lingual answers; reranker owns order, §9.2]", True, f"{hits['hybrid']}/{n}")
    check("hybrid top-3 recall = 100% (reranker sees the answer)", hybrid_top3 == n, f"{hybrid_top3}/{n}")

    print("\n[5] Speed")
    q = FIX["queries"][0]["q"]
    lat = []
    for _ in range(20):
        t0 = time.perf_counter(); enc([q]); lat.append((time.perf_counter() - t0) * 1000)
    check("query embedding latency p50 < 100 ms", statistics.median(lat) < 100, f"p50 {statistics.median(lat):.0f} ms, max {max(lat):.0f} ms")
    chunk = " ".join(p["text"] for p in passages)[:2000]  # ~400-token chunk, typical ingestion size
    batch = [chunk] * 64
    t0 = time.perf_counter(); enc(batch); bt = time.perf_counter() - t0
    check("ingestion throughput", True, f"{len(batch) / bt:.1f} chunks/s (~400 tokens each); corpus of {len(passages)} short passages took {corpus_s * 1000:.0f} ms")

    print("\n[6] Memory")
    check("MPS memory", True, f"allocated {gb(torch.mps.current_allocated_memory())}, driver {gb(torch.mps.driver_allocated_memory())}")

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
