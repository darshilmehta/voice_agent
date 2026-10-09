/**
 * What the chart components share, derived from a `Visual`: the series in their fixed colour slots, grouped by unit
 * (two units never share an axis, so a ₹ crore series and a % series become two small charts), the highlighted
 * positions, and where each number came from (document, page, cell text) for tooltips, labels and the citation popover.
 */

import type { Citation, ProjectDocument } from "../api";
import { toSourceRef, type SourceRef } from "../citations";
import { unitSignature } from "./format";
import type { CellRef, Row, Unit, Visual } from "./types";

/** Categorical slots in the validated palette; a ninth series gets the neutral colour and is never a generated hue. */
export const SERIES_SLOTS = 8;

export interface SeriesView {
  key: string;
  label: string;
  unit: Unit | null;
  calculated: boolean;
  /** Index in the visual's series: the colour follows this, so it never changes when other series come and go. */
  index: number;
  /** 0–7 for the palette's eight hues, null beyond that. */
  slot: number | null;
}

export interface SeriesGroup {
  unit: Unit | null;
  series: SeriesView[];
}

export interface Model {
  visual: Visual;
  xs: string[];
  series: SeriesView[];
  groups: SeriesGroup[];
  /** Positions (row indexes) named by `highlight.x`. */
  highlightX: Set<number>;
  /** Series keys named by `highlight.series`. */
  highlightSeries: Set<string>;
  value: (xi: number, key: string) => number | null;
  cell: (xi: number, key: string) => CellRef | null;
}

const norm = (s: string) => s.trim().toLowerCase();

/** The series a visual draws: its own list, or (a backend that left it out) the keys found in the rows. */
function seriesOf(v: Visual): SeriesView[] {
  let list = v.series;
  if (list.length === 0) {
    const keys: string[] = [];
    for (const r of v.rows) for (const k of Object.keys(r.values)) if (!keys.includes(k)) keys.push(k);
    list = keys.map((k) => ({ key: k, label: k, unit: null, calculated: false }));
  }
  return list.map((s, index) => ({
    key: s.key,
    label: s.label,
    unit: s.unit ?? v.unit,
    calculated: s.calculated,
    index,
    slot: index < SERIES_SLOTS ? index : null,
  }));
}

export function buildModel(v: Visual): Model {
  const series = seriesOf(v);
  const groups: SeriesGroup[] = [];
  const bySig = new Map<string, SeriesGroup>();
  for (const s of series) {
    const sig = unitSignature(s.unit);
    let g = bySig.get(sig);
    if (!g) {
      g = { unit: s.unit, series: [] };
      bySig.set(sig, g);
      groups.push(g);
    }
    g.series.push(s);
  }
  const xs = v.rows.map((r) => r.x);
  const wanted = new Set((v.highlight?.x ?? []).map(norm));
  const highlightX = new Set<number>();
  xs.forEach((x, i) => wanted.has(norm(x)) && highlightX.add(i));
  const rows: Row[] = v.rows;
  return {
    visual: v,
    xs,
    series,
    groups,
    highlightX,
    highlightSeries: new Set(v.highlight?.series ?? []),
    value: (xi, key) => rows[xi]?.values[key] ?? null,
    cell: (xi, key) => rows[xi]?.cells[key] ?? null,
  };
}

/** The CSS variable of a series' colour (the palette tokens in app/globals.css). */
export const seriesColor = (slot: number | null): string => (slot === null ? "var(--viz-other)" : `var(--viz-${slot + 1})`);

// ------------------------------------------------------------------ provenance

export interface CellSource {
  sourceId: string;
  filename: string;
  /** "p.46" style page, or null. */
  page: number | null;
  /** The cell's text as printed in the document. */
  cellText: string;
  /** What the citation popover opens with. */
  ref: SourceRef;
}

/**
 * Resolve a cell to the document it came from. The visual's `sources` are the citations the cell refs point to; the
 * page and cell text come from the cell itself (more exact than the source's page range). When the source is missing
 * the cell still resolves, with a generic file name.
 */
export function resolveCell(visual: Visual, cell: CellRef, docsById: Record<string, ProjectDocument> = {}): CellSource {
  const source: Citation | undefined = visual.sources.find((s) => String(s.source_id ?? "").trim().toUpperCase() === cell.source_id);
  // The popover shows "Cell “4,210”" first and then the cited passage; a markdown table passage still renders as a
  // table, because the text before its first pipe is read as its lead line.
  const passage = typeof source?.snippet === "string" ? source.snippet : "";
  const note = cell.text ? `Cell “${cell.text}”` : "";
  const merged: Citation = {
    ...(source ?? {}),
    source_id: cell.source_id,
    document_id: cell.document_id ?? source?.document_id,
    page_start: cell.page ?? source?.page_start ?? source?.page ?? null,
    page_end: cell.page ?? source?.page_end ?? source?.page ?? null,
    snippet: [note, passage].filter(Boolean).join(passage.includes("|") ? " " : " · ") || undefined,
  };
  const ref = toSourceRef(merged, docsById);
  return { sourceId: cell.source_id, filename: ref.filename, page: ref.pageStart, cellText: cell.text, ref };
}

/** Citations of the visual as popover sources (the footer list). */
export function visualSourceRefs(visual: Visual, docsById: Record<string, ProjectDocument> = {}): SourceRef[] {
  return visual.sources.map((c) => toSourceRef(c, docsById));
}
