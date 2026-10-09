/**
 * Layout math for the charts: nice ticks, bar and stack geometry, waterfall steps, line runs, donut arcs, label
 * fitting and tooltip placement. All of it is pixel layout; none of it produces a number the reader is shown (the
 * frontend never computes a figure: sums, shares and growth rates come from the backend as `Calculation`s).
 *
 * Pure functions with no DOM and no React, so they are checked in Node (frontend/README.md, "Checks").
 */

// ------------------------------------------------------------------ ticks and scales

/** A 1 / 2 / 5 × 10ⁿ step that gives about `count` intervals over `span`. */
export function niceStep(span: number, count: number): number {
  const raw = span / Math.max(1, count);
  if (!(raw > 0) || !Number.isFinite(raw)) return 1;
  const power = Math.floor(Math.log10(raw));
  const base = 10 ** power;
  const error = raw / base;
  const multiple = error >= Math.sqrt(50) ? 10 : error >= Math.sqrt(10) ? 5 : error >= Math.SQRT2 ? 2 : 1;
  return multiple * base;
}

export interface Ticks {
  ticks: number[];
  /** The domain widened to whole steps: [first tick, last tick]. */
  min: number;
  max: number;
  step: number;
}

/** Round to the step's own precision so 0.1 + 0.2 never shows up as a tick. */
const tidy = (value: number, step: number): number => {
  const decimals = Math.max(0, 1 - Math.floor(Math.log10(step)));
  return Number(value.toFixed(Math.min(12, decimals)));
};

/** Round tick values covering [min, max], about `count` intervals apart. */
export function niceTicks(min: number, max: number, count = 4): Ticks {
  let lo = Math.min(min, max);
  let hi = Math.max(min, max);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    lo = 0;
    hi = 1;
  }
  if (lo === hi) {
    // A flat series: give it room either side so the line sits mid-chart.
    const pad = lo === 0 ? 1 : Math.abs(lo) * 0.1;
    lo -= pad;
    hi += pad;
  }
  const step = niceStep(hi - lo, count);
  const first = Math.floor(lo / step + 1e-9) * step;
  const last = Math.ceil(hi / step - 1e-9) * step;
  const ticks: number[] = [];
  for (let i = 0, v = first; v <= last + step * 1e-6 && i < 200; i++, v = first + i * step) ticks.push(tidy(v, step));
  return { ticks, min: ticks[0], max: ticks[ticks.length - 1], step };
}

export interface DomainOptions {
  /** Bars and areas must start at zero; a line only does when zero is near the data. */
  zero: "always" | "auto";
  count?: number;
}

/** The y domain for a set of values (nulls ignored). */
export function niceDomain(values: Array<number | null | undefined>, opts: DomainOptions): Ticks {
  const finite = values.filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  if (finite.length === 0) return niceTicks(0, 1, opts.count ?? 4);
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (opts.zero === "always") {
    min = Math.min(min, 0);
    max = Math.max(max, 0);
  } else if (min >= 0 && min <= 0.6 * max) {
    min = 0;
  } else if (max <= 0 && max >= 0.6 * min) {
    max = 0;
  }
  return niceTicks(min, max, opts.count ?? 4);
}

/** A linear map from [d0, d1] to [r0, r1]; a flat domain maps to the middle of the range. */
export function linearScale(d0: number, d1: number, r0: number, r1: number): (value: number) => number {
  const span = d1 - d0;
  if (span === 0) return () => (r0 + r1) / 2;
  return (v) => r0 + ((v - d0) / span) * (r1 - r0);
}

// ------------------------------------------------------------------ bars

export const MAX_BAR = 24;
export const BAR_GAP = 2;

export interface BarGeometry {
  /** Thickness of one bar, at most `maxBar`. */
  bar: number;
  /** Gap between the bars of a group (the surface colour shows through it). */
  gap: number;
  /** Width of the whole group. */
  group: number;
  /** Where each bar starts, from the slot's start. */
  offsets: number[];
}

/** `k` bars side by side centred in a slot of `slot` pixels: thin bars, a 2 px gap, air around the group. */
export function barGeometry(slot: number, k: number, opts: { maxBar?: number; gap?: number; fill?: number } = {}): BarGeometry {
  const { maxBar = MAX_BAR, gap = BAR_GAP, fill = 0.72 } = opts;
  const count = Math.max(1, k);
  const room = Math.max(1, slot * fill - (count - 1) * gap);
  const bar = Math.max(1, Math.min(maxBar, room / count));
  const group = count * bar + (count - 1) * gap;
  const start = (slot - group) / 2;
  return { bar, gap, group, offsets: Array.from({ length: count }, (_, i) => start + i * (bar + gap)) };
}

export type BarEnd = "top" | "bottom" | "right" | "left" | "none";

/**
 * A bar as path data: a 4 px rounded data end, square at the baseline (and square at both ends for `none`, the inner
 * segments of a stack). `x`, `y`, `w`, `h` are the bar's rectangle; `end` is the side that carries the value.
 */
export function barPath(x: number, y: number, w: number, h: number, end: BarEnd, radius = 4): string {
  const along = end === "top" || end === "bottom" ? h : w;
  const r = end === "none" ? 0 : Math.max(0, Math.min(radius, w / 2, h / 2, along));
  const x1 = x + w;
  const y1 = y + h;
  if (r <= 0 || w <= 0 || h <= 0) return `M${n2(x)} ${n2(y)}H${n2(x1)}V${n2(y1)}H${n2(x)}Z`;
  switch (end) {
    case "top":
      return `M${n2(x)} ${n2(y1)}V${n2(y + r)}Q${n2(x)} ${n2(y)} ${n2(x + r)} ${n2(y)}H${n2(x1 - r)}Q${n2(x1)} ${n2(y)} ${n2(x1)} ${n2(y + r)}V${n2(y1)}Z`;
    case "bottom":
      return `M${n2(x)} ${n2(y)}H${n2(x1)}V${n2(y1 - r)}Q${n2(x1)} ${n2(y1)} ${n2(x1 - r)} ${n2(y1)}H${n2(x + r)}Q${n2(x)} ${n2(y1)} ${n2(x)} ${n2(y1 - r)}Z`;
    case "right":
      return `M${n2(x)} ${n2(y)}H${n2(x1 - r)}Q${n2(x1)} ${n2(y)} ${n2(x1)} ${n2(y + r)}V${n2(y1 - r)}Q${n2(x1)} ${n2(y1)} ${n2(x1 - r)} ${n2(y1)}H${n2(x)}Z`;
    default:
      return `M${n2(x1)} ${n2(y)}H${n2(x + r)}Q${n2(x)} ${n2(y)} ${n2(x)} ${n2(y + r)}V${n2(y1 - r)}Q${n2(x)} ${n2(y1)} ${n2(x + r)} ${n2(y1)}H${n2(x1)}Z`;
  }
}

export interface Segment {
  y0: number;
  y1: number;
}

export interface StackLayout {
  /** `[x][series]`; null for a missing value. Positive values stack up from zero, negative ones down from zero. */
  segments: Array<Array<Segment | null>>;
  min: number;
  max: number;
}

/** Cumulative stacks in value space (layout only: the segments' own values are what is shown). */
export function stackLayout(matrix: Array<Array<number | null | undefined>>): StackLayout {
  let min = 0;
  let max = 0;
  const segments = matrix.map((row) => {
    let up = 0;
    let down = 0;
    return row.map((v): Segment | null => {
      if (typeof v !== "number" || !Number.isFinite(v)) return null;
      if (v >= 0) {
        const seg = { y0: up, y1: up + v };
        up = seg.y1;
        max = Math.max(max, up);
        return seg;
      }
      const seg = { y0: down, y1: down + v };
      down = seg.y1;
      min = Math.min(min, down);
      return seg;
    });
  });
  return { segments, min, max };
}

// ------------------------------------------------------------------ waterfall

export type WaterfallKind = "total" | "up" | "down" | "none";

export interface WaterfallBar {
  kind: WaterfallKind;
  /** Where the bar starts and ends in value space (a total runs from zero). */
  from: number;
  to: number;
  value: number | null;
  /** The level after this step: where the next step starts. */
  level: number;
}

export interface WaterfallLayout {
  bars: WaterfallBar[];
  min: number;
  max: number;
}

/**
 * Start and end bars with increases and decreases floating between them. `total` rows stand on zero and reset the
 * running level to their own value; the others move the level by their value. If the end bar does not equal the level
 * the steps reach, the drawing shows the gap as it is (the backend owns the figures).
 */
export function waterfallLayout(items: Array<{ value: number | null | undefined; total: boolean }>): WaterfallLayout {
  let level = 0;
  let min = 0;
  let max = 0;
  const bars = items.map((it): WaterfallBar => {
    const value = typeof it.value === "number" && Number.isFinite(it.value) ? it.value : null;
    if (value === null) return { kind: "none", from: level, to: level, value: null, level };
    let bar: WaterfallBar;
    if (it.total) {
      level = value;
      bar = { kind: "total", from: 0, to: value, value, level };
    } else {
      const from = level;
      level = from + value;
      bar = { kind: value >= 0 ? "up" : "down", from, to: level, value, level };
    }
    min = Math.min(min, bar.from, bar.to);
    max = Math.max(max, bar.from, bar.to);
    return bar;
  });
  return { bars, min, max };
}

// ------------------------------------------------------------------ lines

export interface Pt {
  x: number;
  y: number;
}

/** Consecutive defined points; a missing value breaks the line instead of drawing through it. */
export function lineRuns(points: Array<Pt | null>): Pt[][] {
  const runs: Pt[][] = [];
  let run: Pt[] = [];
  for (const p of points) {
    if (p) run.push(p);
    else if (run.length) {
      runs.push(run);
      run = [];
    }
  }
  if (run.length) runs.push(run);
  return runs;
}

const n2 = (v: number) => (Math.round(v * 100) / 100).toString();

/** SVG path data for the runs of two or more points (a lone point is drawn as a marker, not a line). */
export function linePath(points: Array<Pt | null>): string {
  return lineRuns(points)
    .filter((r) => r.length >= 2)
    .map((r) => r.map((p, i) => `${i === 0 ? "M" : "L"}${n2(p.x)} ${n2(p.y)}`).join(""))
    .join("");
}

/** The area under a run, down to `baseY`, for the 10 % wash under a single series. */
export function areaPath(points: Array<Pt | null>, baseY: number): string {
  return lineRuns(points)
    .filter((r) => r.length >= 2)
    .map((r) => `${r.map((p, i) => `${i === 0 ? "M" : "L"}${n2(p.x)} ${n2(p.y)}`).join("")}L${n2(r[r.length - 1].x)} ${n2(baseY)}L${n2(r[0].x)} ${n2(baseY)}Z`)
    .join("");
}

/** x positions for dates placed in proportion to time; null when any label is not an ISO date. */
export function proportionalPositions(times: Array<number | null>, r0: number, r1: number): number[] | null {
  if (times.length === 0 || times.some((t) => t === null)) return null;
  const ts = times as number[];
  const lo = Math.min(...ts);
  const hi = Math.max(...ts);
  if (hi === lo) return null;
  return ts.map((t) => r0 + ((t - lo) / (hi - lo)) * (r1 - r0));
}

// ------------------------------------------------------------------ donut

export interface Slice {
  index: number;
  a0: number;
  a1: number;
  /** Angle of the middle of the slice, radians clockwise from 12 o'clock. */
  mid: number;
  d: string;
}

/** Point on a circle, angle clockwise from 12 o'clock. */
export const polar = (cx: number, cy: number, r: number, angle: number): Pt => ({
  x: cx + r * Math.sin(angle),
  y: cy - r * Math.cos(angle),
});

/** A ring segment from angle `a0` to `a1` between radii `r0` (inner) and `r1` (outer). */
export function arcPath(cx: number, cy: number, r0: number, r1: number, a0: number, a1: number): string {
  const sweep = Math.min(a1 - a0, Math.PI * 2 - 1e-4);
  const end = a0 + sweep;
  const large = sweep > Math.PI ? 1 : 0;
  const o0 = polar(cx, cy, r1, a0);
  const o1 = polar(cx, cy, r1, end);
  const i1 = polar(cx, cy, r0, end);
  const i0 = polar(cx, cy, r0, a0);
  return `M${n2(o0.x)} ${n2(o0.y)}A${n2(r1)} ${n2(r1)} 0 ${large} 1 ${n2(o1.x)} ${n2(o1.y)}L${n2(i1.x)} ${n2(i1.y)}A${n2(r0)} ${n2(r0)} 0 ${large} 0 ${n2(i0.x)} ${n2(i0.y)}Z`;
}

/**
 * Slices for the positive values, angles proportional to size (layout only), each shortened by a pixel gap so the
 * surface colour separates neighbours. Values that are missing, zero or negative get no slice (their index is absent).
 */
export function donutSlices(values: Array<number | null | undefined>, cx: number, cy: number, r0: number, r1: number, gapPx = 2): Slice[] {
  const sizes = values.map((v) => (typeof v === "number" && Number.isFinite(v) && v > 0 ? v : 0));
  const total = sizes.reduce((a, b) => a + b, 0);
  if (total <= 0) return [];
  const rMid = (r0 + r1) / 2;
  const pad = gapPx / rMid / 2; // half the gap on each side of a slice
  const slices: Slice[] = [];
  let at = 0;
  sizes.forEach((size, index) => {
    if (size <= 0) return;
    const span = (size / total) * Math.PI * 2;
    const useful = span > pad * 2 + 0.01 && sizes.filter((s) => s > 0).length > 1;
    const a0 = at + (useful ? pad : 0);
    const a1 = at + span - (useful ? pad : 0);
    slices.push({ index, a0, a1, mid: (a0 + a1) / 2, d: arcPath(cx, cy, r0, r1, a0, a1) });
    at += span;
  });
  return slices;
}

// ------------------------------------------------------------------ text fitting

const segmenter: Intl.Segmenter | null =
  typeof Intl !== "undefined" && typeof (Intl as { Segmenter?: unknown }).Segmenter === "function"
    ? new Intl.Segmenter(undefined, { granularity: "grapheme" })
    : null;

/** User-perceived characters: a Devanagari syllable with its vowel signs counts as one. */
export function graphemes(text: string): string[] {
  if (segmenter) return Array.from(segmenter.segment(text), (s) => s.segment);
  return Array.from(text);
}

const ELLIPSIS = "…";

/** At most `maxChars` characters, ending in "…" when cut; never splits a Devanagari conjunct or an emoji. */
export function truncateLabel(text: string, maxChars: number): string {
  const g = graphemes(text);
  if (g.length <= maxChars) return text;
  if (maxChars <= 1) return ELLIPSIS;
  return g.slice(0, maxChars - 1).join("").trimEnd() + ELLIPSIS;
}

export type Measure = (text: string) => number;

const DEVANAGARI = /[ऀ-ॿ]/;

/** A rough text width in pixels when the DOM can't measure it (Node, or before fonts load). */
export function estimateWidth(text: string, fontSize: number): number {
  let em = 0;
  for (const g of graphemes(text)) {
    const c = g.codePointAt(0) ?? 0;
    if (DEVANAGARI.test(g)) em += 0.62 + Math.min(0.5, 0.2 * (Array.from(g).filter((ch) => ch === "्").length));
    else if (c > 0x2e80) em += 1;
    else if (g === " ") em += 0.28;
    else if (/[0-9]/.test(g)) em += 0.58;
    else if (/[iljI.,:;'!|]/.test(g)) em += 0.28;
    else if (/[mwMW@%]/.test(g)) em += 0.86;
    else if (/[A-Z]/.test(g)) em += 0.64;
    else em += 0.54;
  }
  return em * fontSize;
}

/** The longest prefix of `text` that fits in `maxWidth` with a trailing "…" (the text itself when it fits). */
export function fitLabel(text: string, maxWidth: number, measure: Measure): string {
  if (measure(text) <= maxWidth) return text;
  const g = graphemes(text);
  let lo = 0;
  let hi = g.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (measure(g.slice(0, mid).join("").trimEnd() + ELLIPSIS) <= maxWidth) lo = mid;
    else hi = mid - 1;
  }
  return lo === 0 ? ELLIPSIS : g.slice(0, lo).join("").trimEnd() + ELLIPSIS;
}

/** `text` broken at spaces into at most `maxLines` lines that fit, the last one cut with "…" if the text runs on. */
export function wrapLabel(text: string, maxWidth: number, measure: Measure, maxLines = 2): string[] {
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length === 0) return [""];
  const lines: string[] = [];
  let line = "";
  let i = 0;
  while (i < words.length) {
    const next = line ? `${line} ${words[i]}` : words[i];
    if (measure(next) <= maxWidth || !line) {
      line = next;
      i++;
    } else {
      lines.push(line);
      line = "";
      if (lines.length === maxLines - 1) break;
    }
  }
  const rest = [line, ...words.slice(i)].filter(Boolean).join(" ");
  lines.push(rest);
  return lines.slice(0, maxLines).map((l) => fitLabel(l, maxWidth, measure));
}

/**
 * Which labels to show along an axis so none overlap: a regular stride that ends on the last label, plus any
 * `keep` labels (highlighted ones) that still fit beside the rest.
 */
export function pickLabels(positions: number[], widths: number[], gap: number, keep: ReadonlySet<number> = new Set()): number[] {
  const n = positions.length;
  if (n === 0) return [];
  let minStep = Infinity;
  let maxStep = 0;
  for (let i = 1; i < n; i++) {
    const d = Math.abs(positions[i] - positions[i - 1]);
    minStep = Math.min(minStep, d);
    maxStep = Math.max(maxStep, d);
  }
  if (n > 2 && maxStep > minStep * 1.6) {
    // Uneven spacing (dates placed in proportion to time): a stride would be wrong, so pick by what fits where it is,
    // the last label first, then the first, then the ones to keep, then the rest from the left.
    const order = [n - 1, 0, ...keep, ...Array.from({ length: n }, (_, i) => i)];
    const taken: number[] = [];
    for (const i of order) {
      if (i < 0 || i >= n || taken.includes(i)) continue;
      if (taken.every((j) => Math.abs(positions[i] - positions[j]) >= (widths[i] + widths[j]) / 2 + gap)) taken.push(i);
    }
    return taken.sort((a, b) => a - b);
  }
  const widest = Math.max(...widths);
  const stride = n === 1 || !Number.isFinite(minStep) || minStep <= 0 ? 1 : Math.max(1, Math.ceil((widest + gap) / minStep));
  const shown = new Set<number>();
  for (let i = n - 1; i >= 0; i -= stride) shown.add(i);
  for (const k of keep) {
    if (k < 0 || k >= n || shown.has(k)) continue;
    let clear = true;
    for (const j of shown) {
      if (Math.abs(positions[k] - positions[j]) < (widths[k] + widths[j]) / 2 + gap) {
        clear = false;
        break;
      }
    }
    if (clear) shown.add(k);
  }
  return [...shown].sort((a, b) => a - b);
}

// ------------------------------------------------------------------ tooltip and keyboard

export interface Placement {
  left: number;
  top: number;
  side: "right" | "left";
}

/** A tooltip beside its anchor (right, or left when it would overflow), centred on it vertically, kept inside bounds. */
export function placeTooltip(
  anchor: Pt,
  size: { w: number; h: number },
  bounds: { w: number; h: number },
  gap = 12,
): Placement {
  let side: "right" | "left" = "right";
  let left = anchor.x + gap;
  if (left + size.w > bounds.w) {
    side = "left";
    left = anchor.x - gap - size.w;
  }
  left = Math.max(0, Math.min(left, Math.max(0, bounds.w - size.w)));
  const top = Math.max(0, Math.min(anchor.y - size.h / 2, Math.max(0, bounds.h - size.h)));
  return { left, top, side };
}

export interface Cursor {
  xi: number;
  si: number;
}

/**
 * Arrow-key movement over a grid of `nx` positions × `ns` series: left/right change position, up/down change series
 * (up goes to the first series), Home/End jump along the row, PageUp/PageDown to the first/last series. Null when the
 * key isn't one of these or the move would leave the grid.
 */
export function moveCursor(cur: Cursor, key: string, nx: number, ns: number, opts: { vertical?: boolean } = {}): Cursor | null {
  const horizontal = opts.vertical === true; // bars drawn sideways: the categories run down, so up/down changes position
  const clampX = (v: number) => Math.max(0, Math.min(nx - 1, v));
  const clampS = (v: number) => Math.max(0, Math.min(ns - 1, v));
  const along = (delta: number): Cursor | null => {
    const xi = clampX(cur.xi + delta);
    return xi === cur.xi ? null : { xi, si: cur.si };
  };
  const across = (delta: number): Cursor | null => {
    const si = clampS(cur.si + delta);
    return si === cur.si ? null : { xi: cur.xi, si };
  };
  switch (key) {
    case "ArrowRight":
      return horizontal ? across(1) : along(1);
    case "ArrowLeft":
      return horizontal ? across(-1) : along(-1);
    case "ArrowDown":
      return horizontal ? along(1) : across(1);
    case "ArrowUp":
      return horizontal ? along(-1) : across(-1);
    case "Home":
      return cur.xi === 0 ? null : { xi: 0, si: cur.si };
    case "End":
      return cur.xi === nx - 1 ? null : { xi: nx - 1, si: cur.si };
    case "PageUp":
      return cur.si === 0 ? null : { xi: cur.xi, si: 0 };
    case "PageDown":
      return cur.si === ns - 1 ? null : { xi: cur.xi, si: ns - 1 };
    default:
      return null;
  }
}
