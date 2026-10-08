"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { BackendError, fetchHealth, type HealthReport, type HealthStatus } from "@/lib/api";

const STATUS_LABEL: Record<HealthStatus, string> = {
  ok: "OK",
  degraded: "Degraded",
  down: "Down",
  disabled: "Disabled",
  not_implemented: "Not implemented",
};

const CAPABILITY_LABEL: Record<string, string> = {
  llm: "Language model",
  embeddings: "Embeddings",
  reranker: "Reranker",
  vector_store: "Vector store",
  metadata_db: "Database",
  object_store: "File storage",
  ingestion: "Document parsing",
  stt: "Speech to text",
  vad: "Voice detection",
  tts: "Text to speech",
  audio_transport: "Audio transport",
  auth: "Sign-in",
  job_queue: "Job queue",
  session_store: "Session store",
  event_bus: "Event bus",
  web_search: "Web search",
};

const pretty = (capability: string) =>
  CAPABILITY_LABEL[capability] ?? capability.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());

export function StatusPanel({ backendUrl }: { backendUrl: string }) {
  const [report, setReport] = useState<HealthReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [checkedAt, setCheckedAt] = useState<Date | null>(null);
  const inflight = useRef<AbortController | null>(null);

  const check = useCallback(async () => {
    inflight.current?.abort();
    const ctrl = new AbortController();
    inflight.current = ctrl;
    setChecking(true);
    try {
      setReport(await fetchHealth(backendUrl, ctrl.signal));
      setError(null);
    } catch (err) {
      if (ctrl.signal.aborted) return;
      setError(err instanceof BackendError ? err.message : String(err));
    } finally {
      if (!ctrl.signal.aborted) {
        setChecking(false);
        setCheckedAt(new Date());
      }
    }
  }, [backendUrl]);

  useEffect(() => {
    void check();
    return () => inflight.current?.abort();
  }, [check]);

  const counts = report?.providers.reduce<Record<string, number>>((acc, p) => {
    acc[p.status] = (acc[p.status] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <section className="card status" aria-labelledby="status-title" aria-busy={checking}>
      <div className="status-head">
        <div>
          <h2 id="status-title">
            System status{" "}
            {report && (
              <span className="chip" style={{ marginLeft: 6 }}>
                <span className={`dot ${report.status === "ok" ? "ok" : "degraded"}`} aria-hidden />
                {report.status === "ok" ? "All good" : "Needs attention"}
              </span>
            )}
          </h2>
          <p className="sub">
            {report
              ? `${report.providers.length} components · ${Object.entries(counts ?? {})
                  .map(([s, n]) => `${n} ${STATUS_LABEL[s as HealthStatus].toLowerCase()}`)
                  .join(", ")}${report.strict_offline ? " · offline mode" : ""}`
              : error
                ? error
                : "Checking every component…"}
            {checkedAt && ` · checked ${checkedAt.toLocaleTimeString()}`}
          </p>
        </div>
        <button type="button" className="refresh" onClick={() => void check()} disabled={checking}>
          {checking ? "Checking…" : "Recheck"}
        </button>
      </div>

      <div className="providers">
        {report
          ? report.providers.map((p) => (
              <div key={p.capability} className="provider" title={STATUS_LABEL[p.status]}>
                <span className={`dot ${p.status}`} aria-label={STATUS_LABEL[p.status]} role="img" />
                <div className="name">
                  {pretty(p.capability)} <span>· {p.provider}</span>
                </div>
                <div className="detail">{p.detail || STATUS_LABEL[p.status]}</div>
              </div>
            ))
          : !error && Array.from({ length: 8 }, (_, i) => <div key={i} className="skeleton" />)}
      </div>
    </section>
  );
}
