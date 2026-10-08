/**
 * Citations as the UI sees them. Answers mark sources inline as `[S1]` (also `[S1, S2]` or `[S1][S2]`); the
 * message's `citations` (or, while streaming, the `sources` event) say which document, pages and passage each
 * marker means. Older messages hold free-form citation JSON (`document_id`, `page`, `chunk_id`, maybe a name);
 * everything here reads both shapes and never throws on odd input.
 */

import type { Citation, ProjectDocument } from "./api";

export interface SourceRef {
  /** Stable React key. */
  key: string;
  /** "S1", when the citation is numbered. */
  sourceId: string | null;
  /** 1 for "S1". */
  number: number | null;
  documentId: string | null;
  filename: string;
  pageStart: number | null;
  pageEnd: number | null;
  snippet: string | null;
  chunkId: string | null;
}

const str = (v: unknown) => (typeof v === "string" && v.trim() ? v : null);
const int = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);

export const normalizeSourceId = (id: string) => id.trim().toUpperCase();

export function toSourceRef(c: Citation, docsById: Record<string, ProjectDocument> = {}): SourceRef {
  const raw = (c ?? {}) as Record<string, unknown>;
  const sourceId = str(raw.source_id) ? normalizeSourceId(raw.source_id as string) : null;
  const num = sourceId ? Number(/^S(\d+)$/.exec(sourceId)?.[1] ?? NaN) : NaN;
  const documentId = str(raw.document_id);
  const doc = documentId ? docsById[documentId] : undefined;
  const filename =
    str(raw.filename) ?? doc?.filename ?? str(raw.document_name) ?? str(raw.title) ?? str(raw.name) ?? "Document";
  let pageStart = int(raw.page_start) ?? int(raw.page);
  let pageEnd = int(raw.page_end) ?? pageStart;
  if (pageStart === null && pageEnd !== null) pageStart = pageEnd;
  if (pageStart !== null && pageEnd !== null && pageEnd < pageStart) [pageStart, pageEnd] = [pageEnd, pageStart];
  const chunkId = str(raw.chunk_id);
  return {
    key: sourceId ?? `${documentId ?? filename}:${pageStart ?? ""}:${chunkId ?? ""}`,
    sourceId,
    number: Number.isFinite(num) ? num : null,
    documentId,
    filename,
    pageStart,
    pageEnd,
    snippet: str(raw.snippet) ?? str(raw.text),
    chunkId,
  };
}

/** Normalized and de-duplicated (one entry per source id, or per document + page for older citations). */
export function toSourceRefs(citations: unknown, docsById: Record<string, ProjectDocument> = {}): SourceRef[] {
  if (!Array.isArray(citations)) return [];
  const seen = new Set<string>();
  const out: SourceRef[] = [];
  for (const c of citations) {
    if (!c || typeof c !== "object") continue;
    const ref = toSourceRef(c as Citation, docsById);
    if (seen.has(ref.key)) continue;
    seen.add(ref.key);
    out.push(ref);
  }
  return out;
}

/** "p.4", "pp.4–6", or null. */
export function pagesShort(ref: SourceRef): string | null {
  if (ref.pageStart === null) return null;
  return ref.pageEnd !== null && ref.pageEnd !== ref.pageStart ? `pp.${ref.pageStart}–${ref.pageEnd}` : `p.${ref.pageStart}`;
}

/** "Page 4", "Pages 4–6", or null. */
export function pagesLong(ref: SourceRef): string | null {
  if (ref.pageStart === null) return null;
  return ref.pageEnd !== null && ref.pageEnd !== ref.pageStart
    ? `Pages ${ref.pageStart}–${ref.pageEnd}`
    : `Page ${ref.pageStart}`;
}

/** Spoken/accessible description: "Source 1: annual_report.pdf, page 4". */
export function describeSource(ref: SourceRef): string {
  const pages = pagesLong(ref);
  const label = ref.number !== null ? `Source ${ref.number}` : "Source";
  return `${label}: ${ref.filename}${pages ? `, ${pages.toLowerCase()}` : ""}`;
}

// ------------------------------------------------------------------ markers in answer text

export type AnswerPart = { kind: "text"; text: string } | { kind: "cite"; ids: string[]; raw: string };

const MARKER = /\[\s*(S\d+(?:\s*[,;]\s*S?\d+)*)\s*\]/gi;

/** Split answer text into plain text and citation markers. `[S1, 2]` reads as S1 and S2. */
export function splitCitations(text: string): AnswerPart[] {
  const parts: AnswerPart[] = [];
  let last = 0;
  for (const m of text.matchAll(MARKER)) {
    const at = m.index ?? 0;
    if (at > last) parts.push({ kind: "text", text: text.slice(last, at) });
    const ids = m[1]
      .split(/[,;]/)
      .map((s) => s.trim())
      .filter(Boolean)
      .map((s) => normalizeSourceId(/^\d+$/.test(s) ? `S${s}` : s));
    parts.push({ kind: "cite", ids, raw: m[0] });
    last = at + m[0].length;
  }
  if (last < text.length) parts.push({ kind: "text", text: text.slice(last) });
  return parts;
}

/** While streaming, hide a marker that is still being typed ("… margins [S", "[S1, S") so it doesn't flash as text. */
export function withoutPartialMarker(text: string): string {
  return text.replace(/\[\s*(?:S\d*(?:\s*[,;]\s*S?\d*)*)?$/i, "");
}

/** Source ids in the order the text first cites them. */
export function citedIds(text: string): string[] {
  const out: string[] = [];
  for (const part of splitCitations(text)) {
    if (part.kind === "cite") for (const id of part.ids) if (!out.includes(id)) out.push(id);
  }
  return out;
}
