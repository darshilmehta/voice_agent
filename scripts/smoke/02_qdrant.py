# /// script
# requires-python = ">=3.12"
# dependencies = ["qdrant-client>=1.12", "numpy"]
# ///
"""Smoke test 02 — Qdrant.

Verifies the local Qdrant instance supports everything the RAG layer needs,
using synthetic vectors shaped like BGE-M3 output (1024-d dense + sparse
lexical weights over a 250k vocab):

  1. reachable on loopback only, telemetry disabled
  2. named dense + sparse vectors in one collection, payload indexes
  3. dense, sparse and hybrid (RRF) queries rank the right point first
  4. hybrid rescues an exact-identifier query that dense alone misses
  5. metadata filter scopes results to one document
  6. delete-by-document works (document lifecycle)
  7. data survives a container restart
  8. hybrid query latency and container memory

Run:  uv run scripts/smoke/02_qdrant.py [--keep]
"""

import json
import subprocess
import sys
import time
import urllib.request

import numpy as np
from qdrant_client import QdrantClient, models

URL = "http://127.0.0.1:6333"
CONTAINER = "gibberlink-qdrant"
COLLECTION = "smoke_test"
DIM = 1024
VOCAB = 250_002
N_DOCS = 10
CHUNKS_PER_DOC = 100

rng = np.random.default_rng(42)
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


def docker(*args: str) -> str:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=True).stdout.strip()


def unit(v: np.ndarray) -> list[float]:
    return (v / np.linalg.norm(v)).tolist()


def wait_ready(timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{URL}/readyz", timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError("Qdrant not ready")


def make_sparse(n_terms: int = 30) -> models.SparseVector:
    idx = np.sort(rng.choice(VOCAB, size=n_terms, replace=False))
    return models.SparseVector(indices=idx.tolist(), values=rng.uniform(0.05, 0.4, n_terms).round(4).tolist())


def main() -> int:
    keep = "--keep" in sys.argv

    print("\n[1] Instance")
    wait_ready()
    with urllib.request.urlopen(URL, timeout=5) as r:
        version = json.load(r)["version"]
    check("reachable", True, f"Qdrant {version}")
    ports = docker("port", CONTAINER)
    check("bound to 127.0.0.1 only", all(line.split(" -> ")[1].startswith("127.0.0.1") for line in ports.splitlines()), ports.replace("\n", "; "))
    env = docker("inspect", CONTAINER, "--format", "{{json .Config.Env}}")
    check("telemetry disabled", "QDRANT__TELEMETRY_DISABLED=true" in env)

    client = QdrantClient(url=URL)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    print("\n[2] Collection with named dense + sparse vectors")
    client.create_collection(
        COLLECTION,
        vectors_config={"dense": models.VectorParams(size=DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    for field, schema in [("document_id", models.PayloadSchemaType.KEYWORD), ("page_start", models.PayloadSchemaType.INTEGER)]:
        client.create_payload_index(COLLECTION, field_name=field, field_schema=schema)

    dense = rng.standard_normal((N_DOCS * CHUNKS_PER_DOC, DIM)).astype(np.float32)
    sparse = [make_sparse() for _ in range(len(dense))]
    points = [
        models.PointStruct(
            id=i,
            vector={"dense": unit(dense[i]), "sparse": sparse[i]},
            payload={
                "document_id": f"doc{i // CHUNKS_PER_DOC:02d}",
                "chunk_id": f"doc{i // CHUNKS_PER_DOC:02d}_c{i % CHUNKS_PER_DOC:03d}",
                "page_start": i % CHUNKS_PER_DOC // 3 + 1,
                "text": f"synthetic chunk {i}",
            },
        )
        for i in range(len(dense))
    ]
    t = time.perf_counter()
    client.upsert(COLLECTION, points=points, wait=True)
    upsert_s = time.perf_counter() - t
    count = client.count(COLLECTION, exact=True).count
    check("upsert", count == len(points), f"{count} points in {upsert_s:.2f}s")

    print("\n[3] Dense / sparse / hybrid ranking")
    target = 437
    q_dense = unit(dense[target] + 0.6 * rng.standard_normal(DIM))  # paraphrase-like: close, not identical
    t_sp = sparse[target]
    q_sparse = models.SparseVector(indices=t_sp.indices[:8], values=t_sp.values[:8])  # shares a few terms

    def top(**kw) -> list[int]:
        return [p.id for p in client.query_points(COLLECTION, limit=5, **kw).points]

    def hybrid(qd: list[float], qs: models.SparseVector, flt: models.Filter | None = None, limit: int = 5) -> list[int]:
        return [
            p.id
            for p in client.query_points(
                COLLECTION,
                prefetch=[
                    models.Prefetch(query=qd, using="dense", limit=20, filter=flt),
                    models.Prefetch(query=qs, using="sparse", limit=20, filter=flt),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
            ).points
        ]

    d, s, h = top(query=q_dense, using="dense"), top(query=q_sparse, using="sparse"), hybrid(q_dense, q_sparse)
    check("dense ranks target #1", d[0] == target, f"top={d[:3]}")
    check("sparse ranks target #1", s[0] == target, f"top={s[:3]}")
    check("hybrid RRF ranks target #1", h[0] == target, f"top={h[:3]}")

    print("\n[4] Exact-identifier rescue (dense misses, sparse hits)")
    ident = 812
    q_dense_unrelated = unit(rng.standard_normal(DIM))  # embedding has no idea what "SKU-48213" means
    q_ident = models.SparseVector(indices=sparse[ident].indices[:2], values=[1.0, 1.0])
    d2 = top(query=q_dense_unrelated, using="dense")
    h2 = hybrid(q_dense_unrelated, q_ident)
    check("dense alone misses identifier", ident not in d2, f"top={d2[:3]}")
    check("hybrid finds identifier in top 5", ident in h2, f"top={h2}")

    print("\n[5] Document-scoped filter")
    flt = models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchValue(value="doc04"))])
    scoped = client.query_points(
        COLLECTION,
        prefetch=[models.Prefetch(query=q_dense, using="dense", limit=20, filter=flt),
                  models.Prefetch(query=q_sparse, using="sparse", limit=20, filter=flt)],
        query=models.FusionQuery(fusion=models.Fusion.RRF), limit=5, with_payload=True,
    ).points
    docs = {p.payload["document_id"] for p in scoped}
    check("filter restricts to doc04", docs == {"doc04"} and scoped[0].id == target, f"docs={docs}, top={scoped[0].id}")

    print("\n[6] Delete by document")
    client.delete(COLLECTION, points_selector=models.FilterSelector(filter=models.Filter(
        must=[models.FieldCondition(key="document_id", match=models.MatchValue(value="doc09"))])), wait=True)
    after = client.count(COLLECTION, exact=True).count
    check("doc09 chunks removed", after == count - CHUNKS_PER_DOC, f"{count} → {after}")

    print("\n[7] Persistence across restart")
    docker("restart", CONTAINER)
    wait_ready()
    client = QdrantClient(url=URL)
    persisted = client.count(COLLECTION, exact=True).count
    check("count survives restart", persisted == after, f"{persisted} points")
    check("ranking survives restart", hybrid(q_dense, q_sparse)[0] == target)

    print("\n[8] Latency + memory")
    hybrid(q_dense, q_sparse)  # warm
    times = []
    for _ in range(100):
        qd = unit(dense[rng.integers(len(points))] + 0.6 * rng.standard_normal(DIM))
        t = time.perf_counter()
        hybrid(qd, make_sparse(8))
        times.append((time.perf_counter() - t) * 1000)
    p50, p95 = np.percentile(times, [50, 95])
    check("hybrid query p50 < 50 ms", p50 < 50, f"p50={p50:.1f} ms, p95={p95:.1f} ms over 100 queries ({after} points)")
    mem = docker("stats", "--no-stream", "--format", "{{.MemUsage}}", CONTAINER)
    check("container memory", True, mem)

    if not keep:
        client.delete_collection(COLLECTION)
        print("\n  (smoke_test collection deleted; pass --keep to retain)")

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
