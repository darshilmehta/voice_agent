/**
 * The live visual canvas, contract v1 (docs/DESIGN.md §12.1). A `Visual` is what the backend builds from a document's
 * own tables; this frontend only draws it. Every number arrives in the unit's scale with the cell it came from, and
 * every derived number arrives as a `Calculation`: the frontend formats, it never computes.
 *
 * `normalizeVisual` reads a visual defensively (a missing list is empty, an unknown kind falls back to a table), so a
 * backend that is a little ahead of or behind this file never breaks the page.
 */

import type { Citation } from "../api";
import { sortPanels } from "./order";

export const VISUAL_KINDS = [
  "kpi",
  "line",
  "bar",
  "grouped_bar",
  "stacked_bar",
  "waterfall",
  "donut",
  "table",
  "comparison",
  "timeline",
] as const;

export type VisualKind = (typeof VISUAL_KINDS)[number];
export type CanvasLanguage = "en" | "hi";
export type UnitKind = "currency" | "percent" | "count" | "ratio" | "duration" | "none";
export type UnitScale = "crore" | "lakh" | "million" | "billion" | "thousand";
export type CalcOp = "growth" | "cagr" | "diff" | "ratio" | "share" | "sum";
export type DeltaKind = "abs" | "pct" | "pp";
export type XType = "period" | "category" | "date";

export interface Unit {
  kind: UnitKind;
  /** ISO 4217 code ("INR", "USD"); only for currencies. */
  currency: string | null;
  scale: UnitScale | null;
  /** Display label from the backend, e.g. "₹ crore". Used as is where the structured parts say nothing. */
  label: string;
}

/** The exact cell a value was read from. */
export interface CellRef {
  /** "S1": a key into the visual's `sources`. */
  source_id: string;
  document_id: string | null;
  table_id: string | null;
  page: number | null;
  row: number | null;
  col: number | null;
  /** The cell's text as printed in the document ("4,210", "18.4%"). */
  text: string;
}

export interface Calculation {
  label: string;
  op: CalcOp | (string & {});
  value: number;
  unit: Unit | null;
  inputs: CellRef[];
  /** e.g. "(4,210 / 1,980)^(1/4) − 1". Shown beside every calculated number. */
  formula_text: string;
}

export interface Delta {
  value: number;
  kind: DeltaKind;
  calculation: Calculation | null;
}

export interface Tile {
  label: string;
  value: number;
  unit: Unit | null;
  cell: CellRef | null;
  delta: Delta | null;
}

export interface TimelineEvent {
  /** "YYYY-MM-DD", or a free label ("On termination"). */
  date: string;
  label: string;
  detail: string | null;
  cell: CellRef | null;
}

export interface Series {
  key: string;
  label: string;
  unit: Unit | null;
  /** The numbers of this series were derived by the backend's calculator, not read from a cell. */
  calculated: boolean;
}

export interface Row {
  x: string;
  values: Record<string, number | null>;
  cells: Record<string, CellRef | null>;
  /**
   * Waterfall only, and not part of contract v1: a backend may say which rows are totals. Without it the first and the
   * last row are the totals and the rows between are increases and decreases.
   */
  kind?: "total" | "delta";
}

export interface Highlight {
  x: string[];
  series: string[];
  note: string | null;
}

export interface XAxis {
  key: string;
  label: string;
  type: XType;
}

export interface Visual {
  id: string;
  chat_id: string | null;
  project_id: string;
  created_at: string;
  updated_at: string;
  kind: VisualKind;
  /** The kind the backend sent when it isn't one this version draws (it is shown as a table). */
  requested_kind: string | null;
  title: string;
  subtitle: string | null;
  language: CanvasLanguage;
  summary: string;
  unit: Unit | null;
  x: XAxis | null;
  series: Series[];
  rows: Row[];
  tiles: Tile[];
  events: TimelineEvent[];
  highlight: Highlight | null;
  calculations: Calculation[];
  sources: Citation[];
  pinned: boolean;
  position: number;
}

// ------------------------------------------------------------------ events

export type VisualPhase = "preparing" | "ready" | "failed";

/** `event: visual` (SSE) / `{type: "visual"}` (WebSocket). */
export interface VisualEvent {
  type: "visual";
  phase: VisualPhase;
  visualId: string;
  visual: Visual | null;
  detail: string | null;
  turnId: number | null;
}

/** `event: canvas` (SSE) / `{type: "canvas"}` (WebSocket): the whole canvas. */
export interface CanvasSnapshotEvent {
  type: "canvas";
  panels: Visual[];
  turnId: number | null;
}

export type CanvasEvent = VisualEvent | CanvasSnapshotEvent;

export type CanvasOp = "remove" | "pin" | "unpin" | "move";

// ------------------------------------------------------------------ reading

type Json = Record<string, unknown>;

const isObj = (v: unknown): v is Json => !!v && typeof v === "object" && !Array.isArray(v);
const str = (v: unknown): string | null => (typeof v === "string" && v !== "" ? v : null);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);

const UNIT_KINDS: readonly string[] = ["currency", "percent", "count", "ratio", "duration", "none"];
const SCALES: readonly string[] = ["crore", "lakh", "million", "billion", "thousand"];

export function normalizeUnit(raw: unknown): Unit | null {
  if (!isObj(raw)) return null;
  const kind = (UNIT_KINDS.includes(raw.kind as string) ? raw.kind : "none") as UnitKind;
  const scale = SCALES.includes(raw.scale as string) ? (raw.scale as UnitScale) : null;
  return { kind, currency: str(raw.currency), scale, label: typeof raw.label === "string" ? raw.label : "" };
}

export function normalizeCell(raw: unknown): CellRef | null {
  if (!isObj(raw)) return null;
  const sourceId = str(raw.source_id);
  if (!sourceId) return null;
  return {
    source_id: sourceId.trim().toUpperCase(),
    document_id: str(raw.document_id),
    table_id: str(raw.table_id),
    page: num(raw.page),
    row: num(raw.row),
    col: num(raw.col),
    text: typeof raw.text === "string" ? raw.text : "",
  };
}

export function normalizeCalculation(raw: unknown): Calculation | null {
  if (!isObj(raw)) return null;
  const value = num(raw.value);
  if (value === null) return null;
  return {
    label: typeof raw.label === "string" ? raw.label : "",
    op: typeof raw.op === "string" ? raw.op : "diff",
    value,
    unit: normalizeUnit(raw.unit),
    inputs: arr(raw.inputs).map(normalizeCell).filter((c): c is CellRef => c !== null),
    formula_text: typeof raw.formula_text === "string" ? raw.formula_text : "",
  };
}

function normalizeDelta(raw: unknown): Delta | null {
  if (!isObj(raw)) return null;
  const value = num(raw.value);
  if (value === null) return null;
  const kind = (["abs", "pct", "pp"].includes(raw.kind as string) ? raw.kind : "abs") as DeltaKind;
  return { value, kind, calculation: normalizeCalculation(raw.calculation) };
}

function normalizeTile(raw: unknown): Tile | null {
  if (!isObj(raw)) return null;
  const value = num(raw.value);
  if (value === null) return null;
  return {
    label: typeof raw.label === "string" ? raw.label : "",
    value,
    unit: normalizeUnit(raw.unit),
    cell: normalizeCell(raw.cell),
    delta: normalizeDelta(raw.delta),
  };
}

function normalizeEvent(raw: unknown): TimelineEvent | null {
  if (!isObj(raw)) return null;
  const date = typeof raw.date === "string" ? raw.date : "";
  const label = typeof raw.label === "string" ? raw.label : "";
  if (!date && !label) return null;
  return { date, label, detail: str(raw.detail), cell: normalizeCell(raw.cell) };
}

function normalizeRow(raw: unknown): Row | null {
  if (!isObj(raw)) return null;
  const values: Row["values"] = {};
  if (isObj(raw.values)) for (const [k, v] of Object.entries(raw.values)) values[k] = num(v);
  const cells: Row["cells"] = {};
  if (isObj(raw.cells)) for (const [k, v] of Object.entries(raw.cells)) cells[k] = normalizeCell(v);
  const row: Row = { x: typeof raw.x === "string" ? raw.x : String(raw.x ?? ""), values, cells };
  if (raw.kind === "total" || raw.kind === "delta") row.kind = raw.kind;
  return row;
}

function normalizeSeries(raw: unknown, index: number): Series | null {
  if (!isObj(raw)) return null;
  const key = str(raw.key) ?? `s${index + 1}`;
  return { key, label: str(raw.label) ?? key, unit: normalizeUnit(raw.unit), calculated: raw.calculated === true };
}

function normalizeHighlight(raw: unknown): Highlight | null {
  if (!isObj(raw)) return null;
  const x = arr(raw.x).filter((v): v is string => typeof v === "string");
  const series = arr(raw.series).filter((v): v is string => typeof v === "string");
  const note = str(raw.note);
  return x.length || series.length || note ? { x, series, note } : null;
}

/** A visual from the wire, or null when it isn't one (no id). */
export function normalizeVisual(raw: unknown): Visual | null {
  if (!isObj(raw)) return null;
  const id = str(raw.id);
  if (!id) return null;
  const requested = str(raw.kind);
  const known = (VISUAL_KINDS as readonly string[]).includes(requested ?? "");
  const x = isObj(raw.x) ? raw.x : null;
  const xType = (["period", "category", "date"].includes(x?.type as string) ? x?.type : "category") as XType;
  return {
    id,
    chat_id: str(raw.chat_id),
    project_id: str(raw.project_id) ?? "",
    created_at: str(raw.created_at) ?? "",
    updated_at: str(raw.updated_at) ?? "",
    kind: known ? (requested as VisualKind) : "table",
    requested_kind: known ? null : requested,
    title: str(raw.title) ?? "Untitled visual",
    subtitle: str(raw.subtitle),
    language: raw.language === "hi" ? "hi" : "en",
    summary: typeof raw.summary === "string" ? raw.summary : "",
    unit: normalizeUnit(raw.unit),
    x: x ? { key: str(x.key) ?? "x", label: typeof x.label === "string" ? x.label : "", type: xType } : null,
    series: arr(raw.series).map(normalizeSeries).filter((s): s is Series => s !== null),
    rows: arr(raw.rows).map(normalizeRow).filter((r): r is Row => r !== null),
    tiles: arr(raw.tiles).map(normalizeTile).filter((t): t is Tile => t !== null),
    events: arr(raw.events).map(normalizeEvent).filter((e): e is TimelineEvent => e !== null),
    highlight: normalizeHighlight(raw.highlight),
    calculations: arr(raw.calculations).map(normalizeCalculation).filter((c): c is Calculation => c !== null),
    sources: arr(raw.sources).filter(isObj) as Citation[],
    pinned: raw.pinned === true,
    position: num(raw.position) ?? 0,
  };
}

/** Panels in canvas order (`position`, then oldest first); duplicates by id keep the last. */
export function normalizePanels(raw: unknown): Visual[] {
  const byId = new Map<string, Visual>();
  for (const item of arr(raw)) {
    const v = normalizeVisual(item);
    if (v) byId.set(v.id, v);
  }
  return sortPanels([...byId.values()]);
}

export { sortPanels };

/** `event: visual` / `event: canvas` payloads (SSE data, or a WebSocket message), or null for anything else. */
export function parseCanvasEvent(name: string, data: unknown): CanvasEvent | null {
  if (!isObj(data)) return null;
  const turnId = num(data.turn_id);
  if (name === "canvas") {
    return { type: "canvas", panels: normalizePanels(data.panels), turnId };
  }
  if (name === "visual") {
    const visualId = str(data.visual_id) ?? str((isObj(data.visual) ? data.visual : {}).id);
    const phase = data.phase;
    if (!visualId || (phase !== "preparing" && phase !== "ready" && phase !== "failed")) return null;
    return { type: "visual", phase, visualId, visual: normalizeVisual(data.visual), detail: str(data.detail), turnId };
  }
  return null;
}
