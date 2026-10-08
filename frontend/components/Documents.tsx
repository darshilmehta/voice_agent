"use client";

/**
 * A project's documents (docs/DESIGN.md §3.9): drop files anywhere on the card or choose them, watch each upload's
 * progress, then its ingestion (Queued → Processing → Ready with pages and chunks, or Failed with the reason),
 * and delete a document after confirming. Status changes to Ready or Failed are announced politely.
 */

import { useEffect, useRef, useState, type DragEvent } from "react";

import type { Project, ProjectDocument } from "@/lib/api";
import { formatBytes, fullDateTime, plural, shortDate } from "@/lib/format";
import { isIngesting, keys, slotOf, useWorkspace, useWorkspaceActions } from "@/lib/workspace";

import { useEntityActions } from "./Actions";
import { Icon } from "./Icon";
import { BackendDown } from "./States";
import { useUploads, type UploadItem } from "./Uploads";

export const DOCUMENT_STATUS: Record<string, { label: string; tone: string }> = {
  READY: { label: "Ready", tone: "ok" },
  PROCESSING: { label: "Processing", tone: "degraded" },
  PENDING: { label: "Queued", tone: "idle" },
  FAILED: { label: "Failed", tone: "down" },
};

const hasFiles = (e: DragEvent | globalThis.DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes("Files");

export function DocumentsCard({ project }: { project: Project }) {
  const state = useWorkspace();
  const ws = useWorkspaceActions();
  const uploads = useUploads();
  const slot = slotOf(state, keys.docs(project.id));
  const docs = state.documents[project.id] ?? null;
  const items = uploads.itemsFor(project.id);
  const [dragging, setDragging] = useState(false);
  const depth = useRef(0);
  const cardRef = useRef<HTMLElement>(null);
  const zoneRef = useRef<HTMLButtonElement>(null);
  const announcement = useStatusAnnouncements(docs);

  const flashed = new Set(items.filter((i) => i.phase === "duplicate" && i.existing).map((i) => i.existing?.id));
  const compact = (docs?.length ?? 0) > 0 || items.length > 0;
  const pick = () => uploads.pick(project.id);

  // Arriving from "Upload documents" links (…#documents): bring the card into view and focus the drop zone.
  useEffect(() => {
    if (window.location.hash !== "#documents") return;
    cardRef.current?.scrollIntoView({ block: "start" });
    zoneRef.current?.focus({ preventScroll: true });
  }, []);

  // A file dropped next to the card would make the browser navigate to it; swallow those drops.
  useEffect(() => {
    const outside = (e: globalThis.DragEvent) => {
      if (hasFiles(e) && !cardRef.current?.contains(e.target as Node)) e.preventDefault();
    };
    window.addEventListener("dragover", outside);
    window.addEventListener("drop", outside);
    return () => {
      window.removeEventListener("dragover", outside);
      window.removeEventListener("drop", outside);
    };
  }, []);

  const dragProps = {
    onDragEnter: (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth.current += 1;
      setDragging(true);
    },
    onDragOver: (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
    },
    onDragLeave: (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth.current = Math.max(0, depth.current - 1);
      if (depth.current === 0) setDragging(false);
    },
    onDrop: (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth.current = 0;
      setDragging(false);
      if (e.dataTransfer.files.length) uploads.add(project.id, Array.from(e.dataTransfer.files));
    },
  };

  const hint = `${uploads.formats}${uploads.maxMb ? ` · up to ${uploads.maxMb} MB each` : ""}`;

  return (
    <section
      ref={cardRef}
      id="documents"
      className="card docs-card"
      data-dragging={dragging || undefined}
      aria-labelledby="docs-title"
      {...dragProps}
    >
      <div className="card-head">
        <h2 id="docs-title">Documents</h2>
        <button type="button" className="btn btn-sm" onClick={pick}>
          <Icon name="upload" />
          Upload
        </button>
      </div>

      <button
        ref={zoneRef}
        type="button"
        className={compact ? "dropzone compact" : "dropzone"}
        onClick={pick}
        aria-describedby="dropzone-hint"
      >
        <span className="dropzone-icon" aria-hidden>
          <Icon name="upload" size={compact ? 16 : 20} />
        </span>
        <span className="dropzone-text">
          <span className="dropzone-title">
            {dragging ? (
              "Release to upload"
            ) : compact ? (
              <>
                Drop files here or <span className="dropzone-link">choose files</span>
              </>
            ) : (
              "Drop documents here"
            )}
          </span>
          {!compact && <span className="dropzone-sub">{dragging ? " " : "or click to choose files"}</span>}
          <span id="dropzone-hint" className="dropzone-hint">
            {hint}
          </span>
        </span>
      </button>

      {items.length > 0 && (
        <ul className="rows uploads" aria-label="Uploads">
          {items.map((item) => (
            <UploadRow key={item.key} item={item} />
          ))}
        </ul>
      )}

      {docs === null ? (
        slot.status === "error" ? (
          slot.unreachable ? (
            <BackendDown onRetry={() => void ws.loadDocuments(project.id, true)} />
          ) : (
            <p className="form-error">{slot.error}</p>
          )
        ) : (
          <div className="rows" aria-hidden>
            <div className="row-skeleton" />
          </div>
        )
      ) : docs.length === 0 ? (
        items.length === 0 && (
          <p className="note">
            Chats in this project answer from these documents. Each one is read, split into passages and indexed on
            this machine.
          </p>
        )
      ) : (
        <ul className="rows" aria-label="Documents">
          {docs.map((d) => (
            <DocumentRow key={d.id} doc={d} flash={flashed.has(d.id)} />
          ))}
        </ul>
      )}

      <p className="visually-hidden" role="status" aria-live="polite">
        {announcement}
      </p>
    </section>
  );
}

/** "report.pdf is ready: 12 pages." when a document finishes ingesting (not on first load). */
function useStatusAnnouncements(docs: ProjectDocument[] | null): string {
  const previous = useRef<Map<string, string> | null>(null);
  const [message, setMessage] = useState("");
  useEffect(() => {
    if (!docs) return;
    const before = previous.current;
    previous.current = new Map(docs.map((d) => [d.id, d.status]));
    if (!before) return;
    const news: string[] = [];
    for (const d of docs) {
      const was = before.get(d.id);
      if (was === undefined || was === d.status) continue;
      if (d.status === "READY") {
        news.push(`${d.filename} is ready${d.page_count !== null ? `: ${plural(d.page_count, "page")}` : ""}.`);
      } else if (d.status === "FAILED") {
        news.push(`${d.filename} failed${d.error ? `: ${d.error.replace(/[.!]\s*$/, "")}` : ""}.`);
      }
    }
    if (news.length) setMessage(news.join(" "));
  }, [docs]);
  return message;
}

function UploadRow({ item }: { item: UploadItem }) {
  const uploads = useUploads();
  const pct = Math.round(item.progress * 100);
  const active = item.phase === "uploading" || item.phase === "waiting";
  const problem = item.phase === "rejected" || item.phase === "failed";
  const existing = item.existing;

  return (
    <li className="row row-static upload-row" data-phase={item.phase}>
      <Icon name={problem ? "alert" : item.phase === "duplicate" ? "check" : "doc"} className="row-icon" />
      <span className="row-main">
        <span className="row-title">
          <span className="truncate">{item.name}</span>
        </span>
        {active && (
          <>
            <span
              className="progress"
              role="progressbar"
              aria-label={`Uploading ${item.name}`}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={item.phase === "waiting" ? 0 : pct}
            >
              <span className="progress-bar" style={{ width: `${item.phase === "waiting" ? 0 : pct}%` }} />
            </span>
            <span className="row-meta">
              {item.phase === "waiting"
                ? `Waiting · ${formatBytes(item.size)}`
                : pct >= 100
                  ? `Checking the file · ${formatBytes(item.size)}`
                  : `Uploading ${pct}% · ${formatBytes(item.size)}`}
            </span>
          </>
        )}
        {item.phase === "duplicate" && existing && (
          <span className="row-meta">
            Already uploaded{existing.filename !== item.name ? ` as ${existing.filename}` : ""} · nothing new to index
          </span>
        )}
        {item.phase === "rejected" && (
          <span className="row-error">
            Not uploaded. {item.message}
            {item.message && !/[.!?]$/.test(item.message) ? "." : ""}
          </span>
        )}
        {item.phase === "failed" && <span className="row-error">Upload failed: {item.message}</span>}
      </span>
      <span className="row-actions">
        {item.phase === "failed" && (
          <button type="button" className="btn btn-sm" onClick={() => uploads.retry(item.key)}>
            Retry
          </button>
        )}
        {/* Once the whole file is sent the backend may already have stored it: no Cancel that would lie (the
            button keeps its space so the row doesn't shift). */}
        {item.phase === "uploading" && pct >= 100 ? (
          <span className="icon-btn icon-btn-sm" aria-hidden />
        ) : (
          <button
            type="button"
            className="icon-btn icon-btn-sm"
            aria-label={active ? `Cancel uploading ${item.name}` : `Dismiss ${item.name}`}
            title={active ? "Cancel upload" : "Dismiss"}
            onClick={() => (active ? uploads.cancel(item.key) : uploads.dismiss(item.key))}
          >
            <Icon name="close" size={14} />
          </button>
        )}
      </span>
    </li>
  );
}

function DocumentRow({ doc, flash }: { doc: ProjectDocument; flash: boolean }) {
  const actions = useEntityActions();
  const status = DOCUMENT_STATUS[doc.status] ?? { label: doc.status, tone: "idle" };
  const ready = doc.status === "READY";
  const facts = [
    ready && doc.page_count !== null ? plural(doc.page_count, "page") : null,
    ready && doc.chunk_count !== null ? plural(doc.chunk_count, "chunk") : null,
    formatBytes(doc.size_bytes),
  ].filter(Boolean);

  return (
    <li className={flash ? "row row-static doc-row row-flash" : "row row-static doc-row"}>
      <Icon name="doc" className="row-icon" />
      <span className="row-main">
        <span className="row-title">
          <span className="truncate" title={doc.filename}>
            {doc.filename}
          </span>
        </span>
        <span className="row-meta">
          <span className="status-pill" data-status={doc.status}>
            <span className={`dot ${status.tone}${isIngesting(doc) ? " dot-busy" : ""}`} aria-hidden />
            {status.label}
          </span>
          {" · "}
          {facts.join(" · ")}
          {" · "}
          <span title={fullDateTime(doc.created_at)}>added {shortDate(doc.created_at)}</span>
        </span>
        {doc.status === "FAILED" && (
          <span className="row-error">{doc.error || "Ingestion failed without a reason."}</span>
        )}
      </span>
      <button
        type="button"
        className="icon-btn row-more"
        aria-label={`Delete ${doc.filename}`}
        title="Delete document"
        onClick={() => actions.deleteDocument(doc)}
      >
        <Icon name="trash" />
      </button>
    </li>
  );
}
