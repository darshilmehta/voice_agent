/**
 * Citations as the UI sees them. Answers mark sources inline as `[S1]` for a passage of the documents and `[W1]` for a
 * live web result (also `[S1, S2]`, `[S1][W2]`); the message's `citations` (or, while streaming, the `sources` event
 * and the web search's `tool results`) say which document, pages and passage, or which web page, each marker means. A
 * citation without `kind` is a document passage; `kind: "web"` carries `url`, `title`, `site` and `published`. Older
 * messages hold free-form citation JSON (`document_id`, `page`, `chunk_id`, maybe a name); everything here reads all
 * the shapes and never throws on odd input.
 */

import type { Citation, ProjectDocument } from "./api";

export interface SourceRef {
  /** Stable React key. */
  key: string;
  /** A passage of the user's documents, or a live web result. */
  kind: "document" | "web";
  /** "S1" or "W1", when the citation is numbered. */
  sourceId: string | null;
  /** 1 for "S1" and for "W1" (the kind says which). */
  number: number | null;
  documentId: string | null;
  /** Documents: the file name. Web results: the site. */
  filename: string;
  pageStart: number | null;
  pageEnd: number | null;
  snippet: string | null;
  chunkId: string | null;
  /** Web results: where the page is (http or https only; anything else is dropped), its title, site and date. */
  url: string | null;
  title: string | null;
  site: string | null;
  published: string | null;
  /** What the popover says when there is no passage to show (default: none was saved). */
  note?: string;
}

const str = (v: unknown) => (typeof v === "string" && v.trim() ? v : null);
const int = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);

export const normalizeSourceId = (id: string) => id.trim().toUpperCase();

/** The URL when it is an absolute http(s) address, else null: a link from the web is never `javascript:` or `data:`. */
export function safeHttpUrl(value: unknown): string | null {
  const raw = str(value);
  if (!raw) return null;
  try {
    const url = new URL(raw.trim());
    return url.protocol === "http:" || url.protocol === "https:" ? url.href : null;
  } catch {
    return null;
  }
}

/** "reuters.com" for "https://www.reuters.com/markets/…". */
export function hostOf(url: string | null): string | null {
  if (!url) return null;
  try {
    return new URL(url).hostname.replace(/^www\./i, "") || null;
  } catch {
    return null;
  }
}

/** A web result's date: "8 Oct 2026" for an ISO date, the text as given when it isn't one ("2 days ago"), or null. */
export function publishedText(published: string | null): string | null {
  if (!published) return null;
  const text = published.trim();
  if (/^\d{4}-\d{2}-\d{2}(?:[T ]|$)/.test(text)) {
    const d = new Date(/^\d{4}-\d{2}-\d{2}$/.test(text) ? `${text}T00:00:00Z` : text);
    if (!Number.isNaN(d.getTime())) {
      return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
    }
  }
  return text;
}

export function toSourceRef(c: Citation, docsById: Record<string, ProjectDocument> = {}): SourceRef {
  const raw = (c ?? {}) as Record<string, unknown>;
  // A citation without `kind` is a document passage; a web result never looks in the documents (its `document_id` is empty).
  const web = raw.kind === "web";
  const sourceId = str(raw.source_id) ? normalizeSourceId(raw.source_id as string) : null;
  const num = sourceId ? Number(/^[SW](\d+)$/.exec(sourceId)?.[1] ?? NaN) : NaN;
  if (web) {
    const url = safeHttpUrl(raw.url);
    const site = str(raw.site) ?? str(raw.filename) ?? hostOf(url);
    const title = str(raw.title);
    return {
      key: sourceId ?? `web:${url ?? site ?? title ?? ""}`,
      kind: "web",
      sourceId,
      number: Number.isFinite(num) ? num : null,
      documentId: null,
      filename: site ?? "Web",
      pageStart: null,
      pageEnd: null,
      snippet: str(raw.snippet) ?? str(raw.text),
      chunkId: null,
      url,
      title,
      site: site ?? null,
      published: str(raw.published),
    };
  }
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
    kind: "document",
    sourceId,
    number: Number.isFinite(num) ? num : null,
    documentId,
    filename,
    pageStart,
    pageEnd,
    snippet: str(raw.snippet) ?? str(raw.text),
    chunkId,
    url: null,
    title: null,
    site: null,
    published: null,
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

/** What the answer text calls it: "W1" for a web result, "1" for the numbered document passage "S1". */
export function markerOf(ref: SourceRef): string | null {
  if (ref.number === null) return null;
  return ref.kind === "web" ? `W${ref.number}` : String(ref.number);
}

/**
 * Spoken/accessible description: "Source 1: annual_report.pdf, page 4", "Web result 1: Rupee gains, reuters.com,
 * 8 Oct 2026".
 */
export function describeSource(ref: SourceRef): string {
  if (ref.kind === "web") {
    const label = ref.number !== null ? `Web result ${ref.number}` : "Web result";
    const detail = [ref.title, ref.site ?? ref.filename, publishedText(ref.published)].filter(Boolean).join(", ");
    return detail ? `${label}: ${detail}` : label;
  }
  const pages = pagesLong(ref);
  const label = ref.number !== null ? `Source ${ref.number}` : "Source";
  return `${label}: ${ref.filename}${pages ? `, ${pages.toLowerCase()}` : ""}`;
}

/** Documents first, then web results; each in numbered order (un-numbered last, in the order they came). */
export function sortSources<T extends SourceRef>(refs: readonly T[]): T[] {
  const rank = (r: SourceRef) => (r.kind === "web" ? 1 : 0);
  return refs
    .map((r, i) => ({ r, i }))
    .sort((a, b) => rank(a.r) - rank(b.r) || (a.r.number ?? Infinity) - (b.r.number ?? Infinity) || a.i - b.i)
    .map((x) => x.r);
}

// ------------------------------------------------------------------ table snippets

/** A markdown table found inside a citation snippet, ready to render as a compact table. */
export interface TableSnippet {
  /** Prose that came before the table, if any. */
  lead: string;
  header: string[];
  rows: string[][];
  /** The snippet's length limit cut the table short (rows or part of a row are missing). */
  truncated: boolean;
}

const DELIMITER_CELL = /^:?-{2,}:?$/;
const MAX_TABLE_ROWS = 6;

/**
 * Reads a markdown table out of a snippet, or returns null when the snippet isn't one (then show it as plain text).
 *
 * The backend flattens snippets to one line (whitespace runs, newlines included, become a single space and the text
 * is cut at ~300 characters), so rows can't be told apart by line breaks: `| Metric | FY23 | | EBITDA | 16.9% |`.
 * The delimiter row (`|---|---|`) gives the column count; the header is the cells before it, and every complete
 * row is that many cells followed by the pipe that closes it. A row cut off by the length limit is dropped and the
 * table is marked truncated. Snippets that still hold line breaks parse the same way.
 */
export function parseTableSnippet(text: string): TableSnippet | null {
  const flat = text.replace(/\s+/g, " ").trim();
  const first = flat.indexOf("|");
  if (first < 0) return null;
  const lead = flat.slice(0, first).trim();
  // Split on unescaped pipes; the leading pipe leaves an empty first piece.
  const cells = flat
    .slice(first)
    .split(/(?<!\\)\|/)
    .slice(1)
    .map((c) =>
      c
        .replace(/\\\|/g, "|")
        .replace(/<br\s*\/?>/gi, " ")
        .replace(/\*\*|__/g, "")
        .trim(),
    );

  let delimiterAt = -1;
  let columns = 0;
  for (let i = 0; i < cells.length; i++) {
    if (!DELIMITER_CELL.test(cells[i])) continue;
    let end = i;
    while (end < cells.length && DELIMITER_CELL.test(cells[end])) end++;
    if (end - i >= 2) {
      delimiterAt = i;
      columns = end - i;
      break;
    }
    i = end;
  }
  // The header is `columns` cells, then the pipe pair that ends its row, then the delimiter row.
  if (delimiterAt < 0 || delimiterAt !== columns + 1 || cells[delimiterAt - 1] !== "") return null;

  const header = cells.slice(0, columns);
  const rows: string[][] = [];
  let at = delimiterAt + columns + 1;
  while (at + columns < cells.length) {
    rows.push(cells.slice(at, at + columns));
    at += columns + 1;
  }
  const leftover = cells.slice(at).some((c) => c !== "");
  const truncated = leftover || flat.endsWith("…") || rows.length > MAX_TABLE_ROWS;
  return { lead, header, rows: rows.slice(0, MAX_TABLE_ROWS), truncated };
}

/** Numbers, percentages and currency amounts line up to the right. */
export function isNumericCell(cell: string): boolean {
  return /^[(+\-−–]?\s*[₹$€£]?\s*\d[\d,.\s]*\s*(%|x|×|cr|crore|bn|m|k)?\)?$/i.test(cell.trim());
}

// ------------------------------------------------------------------ markers in answer text

export type AnswerPart = { kind: "text"; text: string } | { kind: "cite"; ids: string[]; raw: string };

const MARKER = /\[\s*([SW]\d+(?:\s*[,;]\s*[SW]?\d+)*)\s*\]/gi;

/**
 * Split answer text into plain text and citation markers. `[S1, 2]` reads as S1 and S2, `[W1, 2]` as W1 and W2 (a bare
 * number goes with the kind before it), `[S1, W2]` as one of each.
 */
export function splitCitations(text: string): AnswerPart[] {
  const parts: AnswerPart[] = [];
  let last = 0;
  for (const m of text.matchAll(MARKER)) {
    const at = m.index ?? 0;
    if (at > last) parts.push({ kind: "text", text: text.slice(last, at) });
    let prefix = "S";
    const ids = m[1]
      .split(/[,;]/)
      .map((s) => s.trim())
      .filter(Boolean)
      .map((s) => {
        if (/^\d+$/.test(s)) return `${prefix}${s}`;
        prefix = s[0].toUpperCase();
        return normalizeSourceId(s);
      });
    parts.push({ kind: "cite", ids, raw: m[0] });
    last = at + m[0].length;
  }
  if (last < text.length) parts.push({ kind: "text", text: text.slice(last) });
  return parts;
}

/** While streaming, hide a marker that is still being typed ("… margins [S", "[W1, W", "[") so it doesn't flash as text. */
export function withoutPartialMarker(text: string): string {
  return text.replace(/\[\s*(?:[SW]\d*(?:\s*[,;]\s*[SW]?\d*)*)?$/i, "");
}

/** Source ids in the order the text first cites them. */
export function citedIds(text: string): string[] {
  const out: string[] = [];
  for (const part of splitCitations(text)) {
    if (part.kind === "cite") for (const id of part.ids) if (!out.includes(id)) out.push(id);
  }
  return out;
}
