"use client";

/** Pieces the drawn charts share: the width-measuring box, the legend, and how a point is described. */

import { useRef, type ReactNode } from "react";

import { formatDate, formatValue, isIsoDate, unitCaption } from "@/lib/canvas/format";
import { seriesColor, type Model, type SeriesView } from "@/lib/canvas/model";
import type { CellRef, Calculation, Visual } from "@/lib/canvas/types";

import type { Described, Tip, TipRow } from "./ChartFrame";
import { useElementWidth } from "./measure";
import { CalcBadge, usePanel, type PanelApi } from "./panel-context";

/** Measures its own width and draws its chart at exactly that width (so SVG pixels are CSS pixels). */
export function ChartBox({ children, className }: { children: (width: number) => ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const width = useElementWidth(ref);
  return (
    <div ref={ref} className={className ? `cv-box ${className}` : "cv-box"}>
      {width > 0 ? children(width) : null}
    </div>
  );
}

export interface LegendItem {
  key: string;
  label: string;
  color: string;
  shape: "line" | "bar";
  calculated?: boolean;
  /** Highlighted by the visual: drawn in bold. */
  strong?: boolean;
}

/** Always shown for two or more series (the dependable identity channel; direct labels only supplement it). */
export function Legend({ items, label }: { items: LegendItem[]; label?: string }) {
  if (items.length < 2) return null;
  return (
    <ul className="cv-legend" aria-label={label}>
      {items.map((it) => (
        <li key={it.key} data-strong={it.strong || undefined}>
          <span className="cv-key" data-shape={it.shape} style={{ background: it.color }} aria-hidden="true" />
          <span className="cv-legend-label">{it.label}</span>
          {it.calculated && <CalcBadge />}
        </li>
      ))}
    </ul>
  );
}

export function seriesLegend(model: Model, shape: "line" | "bar"): LegendItem[] {
  return model.series.map((s) => ({
    key: s.key,
    label: s.label,
    color: seriesColor(s.slot),
    shape,
    calculated: s.calculated,
    strong: model.highlightSeries.has(s.key) || undefined,
  }));
}

/** A caption above one axis: the unit ("₹ crore", "%"), in muted text. */
export const axisCaption = (unit: Parameters<typeof unitCaption>[0], language: "en" | "hi") => unitCaption(unit, language);

/** "FY23" for a period or category; a date axis writes ISO dates as "15 Mar 2025". */
export function xText(visual: Visual, x: string): string {
  return visual.x?.type === "date" && isIsoDate(x) ? formatDate(x, visual.language) : x;
}

/** The calculation that stands behind a calculated series (matched by name), for its formula in the tooltip. */
export function calculationFor(visual: Visual, s: SeriesView, x?: string): Calculation | null {
  if (!s.calculated) return null;
  const n = (t: string) => t.trim().toLowerCase();
  const byName = visual.calculations.filter((c) => n(c.label).includes(n(s.label)) || n(s.label).includes(n(c.label)));
  if (byName.length === 0) return null;
  if (x) {
    const exact = byName.find((c) => n(c.label).includes(n(x)));
    if (exact) return exact;
  }
  return byName[0];
}

export interface PointFacts {
  value: number | null;
  /** "₹4,210 crore", or "—". */
  text: string;
  /** "₹4,210 crore", or "no value", with the sign spelled out. */
  spoken: string;
  cell: CellRef | null;
}

export function pointFacts(model: Model, xi: number, s: SeriesView): PointFacts {
  const value = model.value(xi, s.key);
  const lang = model.visual.language;
  return {
    value,
    text: formatValue(value, s.unit, lang),
    spoken: formatValue(value, s.unit, lang, { spoken: true }),
    cell: model.cell(xi, s.key),
  };
}

/**
 * The description of one point: its accessible name (position, series, value, whether it is calculated, where it came
 * from) and its tooltip (every series at that position, with the active one's source and formula).
 */
export function describePoint(
  model: Model,
  panel: PanelApi,
  xi: number,
  si: number,
  opts: { shape: "line" | "bar" | "dot"; title?: string; rows?: SeriesView[] } = { shape: "bar" },
): Described {
  const { labels } = panel;
  const s = model.series[si];
  const lang = model.visual.language;
  const xLabel = opts.title ?? xText(model.visual, model.xs[xi] ?? "");
  const facts = pointFacts(model, xi, s);
  const src = facts.cell ? panel.resolve(facts.cell) : null;
  const calc = calculationFor(model.visual, s, model.xs[xi]);
  const rowsOf = opts.rows ?? model.series;
  const rows: TipRow[] = rowsOf.map((r) => ({
    color: seriesColor(r.slot),
    shape: opts.shape,
    label: r.label,
    value: model.value(xi, r.key) === null ? labels.notReported : formatValue(model.value(xi, r.key), r.unit, lang),
    calculated: r.calculated,
    active: r.index === si,
  }));
  const tip: Tip = {
    title: xLabel,
    rows,
    source: src ? { filename: src.filename, page: src.page, cellText: src.cellText } : null,
    formula: s.calculated ? (calc?.formula_text ?? null) : null,
  };
  const parts = [
    `${xLabel}, ${s.label}: ${facts.spoken}`,
    s.calculated ? labels.calculated : null,
    src ? `${labels.source}: ${src.filename}${src.page !== null ? `, ${labels.page} ${src.page}` : ""}` : null,
  ].filter(Boolean);
  return { aria: parts.join(". "), tip, cell: facts.cell };
}

/** Where the numbers sit in the visual's own unit: the first series' unit, then the visual's. */
export const primaryUnit = (model: Model) => model.series[0]?.unit ?? model.visual.unit;

/** An empty chart says so instead of drawing nothing. */
export function EmptyNote() {
  const { labels } = usePanel();
  return <p className="cv-empty">{labels.empty}</p>;
}
