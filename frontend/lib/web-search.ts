/**
 * Live web search as the UI sees it (docs/DESIGN.md §3.7). The backend reports a turn's search with `tool` events
 * (SSE `event: tool`, WebSocket `{"type":"tool","turn_id":N,…}`), both carrying the same fields:
 *
 *   start    { name: "web_search", phase: "start", query }           the query that is leaving the machine
 *   results  { …, sources: [web citation, …], elapsed_ms }           new results, numbered W1, W2… in arrival order
 *   done | timeout | failed  { …, count, elapsed_ms, detail? }       the search ended (`detail` on failed only)
 *
 * A terminal event can arrive in the middle of the answer's deltas, and results can arrive after the answer started
 * (a continuation that cites `[W3]`). Only the end of the turn ends a turn, so the "searching" state is cleared by
 * more than the terminal event: the error and cut paths may send none (`endSearch`, called on `sources`, the saved
 * message, an error, a cut, a stop or a new turn). Everything here is pure; the transports (lib/chat-turns.ts,
 * lib/voice/session.ts) keep one `WebTurn` per turn.
 */

import type { Citation } from "./api";
import { normalizeSourceId } from "./citations";

export type ToolPhase = "start" | "results" | "done" | "timeout" | "failed";

const PHASES: ReadonlySet<string> = new Set<ToolPhase>(["start", "results", "done", "timeout", "failed"]);

/** One `tool` event, read without trusting the wire. */
export interface ToolEvent {
  phase: ToolPhase;
  query: string;
  /** `results`: the new web citations (never renumbered). */
  sources: Citation[];
  /** Terminal phases: how many results the search returned. */
  count: number | null;
  elapsedMs: number | null;
  /** `failed`: why. */
  detail: string | null;
}

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const text = (v: unknown): string | null => (typeof v === "string" && v.trim() ? v.trim() : null);

/** The event, or null when it isn't a (well-formed) web search event. Other tools are ignored, not fatal. */
export function parseTool(value: unknown): ToolEvent | null {
  if (!value || typeof value !== "object") return null;
  const v = value as Record<string, unknown>;
  if (v.name !== undefined && v.name !== "web_search") return null;
  if (typeof v.phase !== "string" || !PHASES.has(v.phase)) return null;
  const sources = Array.isArray(v.sources)
    ? v.sources.filter((s): s is Citation => !!s && typeof s === "object" && !Array.isArray(s))
    : [];
  return {
    phase: v.phase as ToolPhase,
    query: text(v.query) ?? "",
    sources,
    count: num(v.count),
    elapsedMs: num(v.elapsed_ms),
    detail: text(v.detail),
  };
}

/**
 * `searching`: started, no end yet. `done` / `timeout` / `failed`: the backend said how it ended. `ended`: nothing is
 * searching any more, but the backend never said how (the answer began, the turn was cut or failed).
 */
export type SearchStatus = "searching" | "done" | "timeout" | "failed" | "ended";

export interface WebSearchState {
  /** What was searched: exactly what left the machine. */
  query: string;
  status: SearchStatus;
  count: number | null;
  elapsedMs: number | null;
  detail: string | null;
}

/** A turn's web side: its search and every web result that arrived for it, for `[W#]` markers. */
export interface WebTurn {
  search: WebSearchState | null;
  /** Web citations from `tool results`, in arrival order, merged by source id. */
  sources: Citation[];
}

export const NO_WEB: WebTurn = { search: null, sources: [] };

/** Add citations by source id; a repeated id replaces the earlier entry in place, so numbering never moves. */
export function mergeWebSources(have: Citation[], add: Citation[]): Citation[] {
  if (add.length === 0) return have;
  const out = have.slice();
  for (const c of add) {
    const id = typeof c.source_id === "string" && c.source_id.trim() ? normalizeSourceId(c.source_id) : null;
    if (!id) continue;
    const at = out.findIndex((o) => typeof o.source_id === "string" && normalizeSourceId(o.source_id) === id);
    if (at >= 0) out[at] = c;
    else out.push(c);
  }
  return out;
}

/**
 * Apply a `tool` event. Results are always merged (a `[W3]` marker must resolve, whatever else is hidden); the search
 * state, which drives the badge and the note, is only kept when `show` is true (`features.web_search`).
 */
export function applyTool(web: WebTurn, ev: ToolEvent, show: boolean): WebTurn {
  const sources = ev.phase === "results" ? mergeWebSources(web.sources, ev.sources) : web.sources;
  if (!show) return sources === web.sources ? web : { ...web, sources };
  const cur = web.search;
  let search: WebSearchState;
  switch (ev.phase) {
    case "start":
      search = { query: ev.query, status: "searching", count: null, elapsedMs: null, detail: null };
      break;
    case "results":
      // Results say the search ran; they never bring the badge back once it has gone.
      search = cur
        ? { ...cur, query: cur.query || ev.query, elapsedMs: ev.elapsedMs ?? cur.elapsedMs }
        : { query: ev.query, status: "ended", count: null, elapsedMs: ev.elapsedMs, detail: null };
      break;
    default:
      search = {
        query: cur?.query || ev.query,
        status: ev.phase,
        count: ev.count ?? cur?.count ?? null,
        elapsedMs: ev.elapsedMs ?? cur?.elapsedMs ?? null,
        detail: ev.detail,
      };
  }
  return { search, sources };
}

/** Nothing is searching any more (sources arrived, the answer was saved, an error, a cut, a stop). */
export function endSearch(web: WebTurn): WebTurn {
  return web.search?.status === "searching" ? { ...web, search: { ...web.search, status: "ended" } } : web;
}

export const isSearching = (web: WebTurn): boolean => web.search?.status === "searching";

/** The search the saved message's route recorded (`route.web_search`), when it says what was searched. */
export function searchOfRoute(route: Record<string, unknown> | null | undefined): WebSearchState | null {
  const rec = route?.web_search;
  if (!rec || typeof rec !== "object") return null;
  const r = rec as Record<string, unknown>;
  const query = text(r.query);
  if (!query) return null;
  const status: SearchStatus = r.status === "done" || r.status === "timeout" || r.status === "failed" ? r.status : "ended";
  return { query, status, count: num(r.results), elapsedMs: null, detail: text(r.error) };
}

/** "Searched the web for “…”", and how it ended when that is worth saying. Quiet: counts are in the hover text. */
export function searchNote(s: WebSearchState): { lead: string; query: string; tail: string | null } {
  const query = s.query;
  switch (s.status) {
    case "timeout":
      return { lead: query ? "Searched the web for" : "Searched the web", query, tail: "timed out" };
    case "failed":
      return { lead: query ? "Web search failed for" : "Web search failed", query, tail: null };
    case "done":
      return { lead: query ? "Searched the web for" : "Searched the web", query, tail: s.count === 0 ? "no results" : null };
    default:
      return { lead: query ? "Searched the web for" : "Searched the web", query, tail: null };
  }
}

/** Hover text for the note: how many results and how long it took. */
export function searchDetail(s: WebSearchState): string | undefined {
  const parts: string[] = [];
  if (s.count !== null) parts.push(`${s.count} ${s.count === 1 ? "result" : "results"}`);
  if (s.elapsedMs !== null) parts.push(`${(s.elapsedMs / 1000).toFixed(1)} s`);
  if (s.status === "failed" && s.detail) parts.push(s.detail);
  return parts.length ? parts.join(" · ") : undefined;
}
