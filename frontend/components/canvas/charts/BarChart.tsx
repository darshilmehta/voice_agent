"use client";

/**
 * Bar, grouped bar and stacked bar. Thin bars (24 px at most) with a 4 px rounded data end and a square baseline, a
 * 2 px surface gap between touching marks, values at the tips only where they fit (never a number on every bar),
 * recessive grid. Nominal categories take one colour; separate series take the palette in its fixed order. Few
 * categories stand up as columns, many (or long names, or a long list) lie down as rows, and a list longer than twelve
 * folds to its first twelve with "Show all". Series in different units never share an axis: one chart per unit.
 */

import { useCallback, useMemo, useState } from "react";

import { formatCell, formatTick, unitCaption } from "@/lib/canvas/format";
import { seriesColor, type Model, type SeriesGroup } from "@/lib/canvas/model";
import {
  barGeometry,
  barPath,
  fitLabel,
  linearScale,
  niceDomain,
  niceTicks,
  pickLabels,
  stackLayout,
  wrapLabel,
  type BarEnd,
} from "@/lib/canvas/scales";

import { ChartFrame, type HitBox } from "../ChartFrame";
import { ChartBox, Legend, describePoint, seriesLegend, xText } from "../chart-common";
import { useMeasure } from "../measure";
import { usePanel } from "../panel-context";

export type BarVariant = "bar" | "grouped" | "stacked";

const ROW_LIMIT = 12;

export function BarChart({ model, variant, summaryId }: { model: Model; variant: BarVariant; summaryId: string }) {
  const { labels } = usePanel();
  // A plain bar chart has one series; a backend that sends several means them side by side.
  const effective: BarVariant = variant === "bar" && model.series.length > 1 ? "grouped" : variant;
  const groups: SeriesGroup[] = effective === "bar" ? [{ unit: model.series[0]?.unit ?? null, series: model.series.slice(0, 1) }] : model.groups;
  return (
    <>
      <Legend items={seriesLegend(model, "bar")} label={labels.series} />
      {groups.map((g, gi) => (
        <ChartBox key={gi} className={gi > 0 ? "cv-box-next" : undefined}>
          {(width) => <BarGroup model={model} variant={effective} group={g} width={width} summaryId={summaryId} />}
        </ChartBox>
      ))}
    </>
  );
}

interface Bar {
  xi: number;
  /** Position among the group's series. */
  s: number;
  value: number;
  path: string;
  rect: { x: number; y: number; w: number; h: number };
  color: string;
  /** The tip of the bar (where its value label goes and the tooltip points). */
  tip: { x: number; y: number };
  sign: 1 | -1;
}

function BarGroup({ model, variant, group, width, summaryId }: { model: Model; variant: BarVariant; group: SeriesGroup; width: number; summaryId: string }) {
  const panel = usePanel();
  const { visual, labels } = panel;
  const lang = visual.language;
  const measure = useMeasure(12);
  const strong = useMeasure(12, 600);
  const [expanded, setExpanded] = useState(false);
  const series = group.series;
  const k = series.length;
  const stacked = variant === "stacked";
  const narrow = width < 480;
  const nAll = model.xs.length;

  const xTexts = useMemo(() => model.xs.map((x) => xText(visual, x)), [model, visual]);
  const timeAxis = visual.x?.type === "period" || visual.x?.type === "date";
  const widest = Math.max(0, ...xTexts.map((t) => measure(t)));
  const rows = !timeAxis && (nAll > 6 || widest > (width - 70) / Math.max(1, nAll) - 6);
  const m = rows && !expanded ? Math.min(nAll, ROW_LIMIT) : nAll;
  const folded = rows && nAll > ROW_LIMIT;

  const caption = unitCaption(group.unit, lang);
  const emphasis = k === 1 && model.highlightX.size > 0;
  const dimOthers = model.highlightSeries.size > 0;

  // Values in the visible positions, per series (nulls stay nulls).
  const matrix = useMemo(() => Array.from({ length: m }, (_, xi) => series.map((s) => model.value(xi, s.key))), [model, series, m]);
  const stack = useMemo(() => (stacked ? stackLayout(matrix) : null), [stacked, matrix]);
  const domain = useMemo(() => {
    if (stack) return niceTicks(stack.min, stack.max, narrow ? 3 : 4);
    return niceDomain(matrix.flat(), { zero: "always", count: narrow ? 3 : 4 });
  }, [stack, matrix, narrow]);

  const tickText = domain.ticks.map((t) => formatTick(t, group.unit));
  const tickW = Math.max(...tickText.map((t) => measure(t)), 24);

  // ---- geometry
  const CAP = 28;
  const plotH = narrow ? 150 : 188;
  const rowSlot = k === 1 || stacked ? 28 : Math.max(30, k * 14 + (k - 1) * 2 + 12);
  // The highlighted label is bold, so it is measured bold.
  const labelFont = (xi: number) => (model.highlightX.has(xi) ? strong : measure);
  const labelW = Math.min(width * 0.4, Math.max(48, Math.max(0, ...xTexts.slice(0, m).map((t, xi) => labelFont(xi)(t))) + 4));
  const x0 = rows ? labelW + 12 : tickW + 14;
  const catLabels = rows ? xTexts.slice(0, m).map((t, xi) => fitLabel(t, labelW, labelFont(xi))) : xTexts;

  // Values written at the tips: which ones, decided from what fits.
  // Tip labels are the number alone: the axis caption says "₹ crore", and a bar is only 24 px wide.
  const labelTexts = useMemo(
    () => matrix.map((row) => row.map((v, s) => (v === null ? "" : formatCell(v, series[s].unit)))),
    [matrix, series],
  );
  const slotCol = (width - x0 - 14) / Math.max(1, m);
  const wantLabels = useMemo(() => {
    const want = new Set<string>();
    if (stacked) return want;
    const all: Array<[number, number, number]> = [];
    matrix.forEach((row, xi) => row.forEach((v, s) => v !== null && all.push([xi, s, v])));
    const maxLabel = Math.max(0, ...all.map(([xi, s]) => strong(labelTexts[xi][s])));
    const fitsAll = rows ? all.length <= 8 : k === 1 ? all.length <= 8 && maxLabel + 4 <= slotCol : false;
    if (fitsAll) {
      all.forEach(([xi, s]) => want.add(`${xi}:${s}`));
      return want;
    }
    // Selective: what the story is about (highlighted positions), then the extremes of a single series.
    model.highlightX.forEach((xi) => xi < m && series.forEach((_, s) => matrix[xi][s] !== null && want.add(`${xi}:${s}`)));
    if (k === 1 && all.length > 0) {
      const vals = all.map(([, , v]) => v);
      const hi = all.find(([, , v]) => v === Math.max(...vals));
      const lo = all.find(([, , v]) => v === Math.min(...vals));
      if (hi) want.add(`${hi[0]}:0`);
      if (lo) want.add(`${lo[0]}:0`);
      if (!rows && all.length > 0) want.add(`${all[all.length - 1][0]}:0`);
    }
    if (!rows && k > 1 && want.size === 0 && all.length > 0) {
      // Grouped columns: the last position, when its labels fit side by side.
      const last = m - 1;
      if (series.every((_, s) => matrix[last][s] === null || strong(labelTexts[last][s]) * k + 4 * k <= slotCol)) {
        series.forEach((_, s) => matrix[last][s] !== null && want.add(`${last}:${s}`));
      }
    }
    return want;
  }, [matrix, labelTexts, stacked, rows, k, slotCol, model.highlightX, m, series, strong]);

  const rightPad = rows
    ? 14 + Math.max(0, ...[...wantLabels].map((key) => {
        const [xi, s] = key.split(":").map(Number);
        return strong(labelTexts[xi][s]) + 8;
      }))
    : 14;
  const x1 = width - rightPad;
  const axisH = rows ? 26 : 38;
  const plotWidth = x1 - x0;
  const height = CAP + (rows ? m * rowSlot : plotH) + axisH;
  const y0 = CAP;
  const y1 = y0 + (rows ? m * rowSlot : plotH);

  // Room for the value written at a bar's tip: above the highest, and below the lowest when a bar goes negative.
  const hasNegative = matrix.some((row) => row.some((v) => v !== null && v < 0));
  const padEnd = 18;
  const scale = rows
    ? linearScale(domain.min, domain.max, x0 + (hasNegative ? padEnd : 0), x1)
    : linearScale(domain.min, domain.max, y1 - (hasNegative ? padEnd : 0), y0 + 14);
  const baseline = scale(Math.max(domain.min, Math.min(0, domain.max)));
  const slot = rows ? rowSlot : plotWidth / Math.max(1, m);
  const geom = barGeometry(slot, stacked ? 1 : k, { maxBar: rows ? 20 : 24 });

  // ---- bars
  const bars: Bar[] = useMemo(() => {
    const out: Bar[] = [];
    for (let xi = 0; xi < m; xi++) {
      const lastUp = stack ? series.map((_, s) => s).filter((s) => stack.segments[xi][s] && (matrix[xi][s] as number) >= 0).pop() : -1;
      const lastDown = stack ? series.map((_, s) => s).filter((s) => stack.segments[xi][s] && (matrix[xi][s] as number) < 0).pop() : -1;
      series.forEach((srs, s) => {
        const v = matrix[xi][s];
        if (v === null) return;
        const color = emphasis && !model.highlightX.has(xi) ? "var(--viz-other)" : seriesColor(srs.slot);
        const catStart = (rows ? y0 : x0) + xi * slot;
        const off = geom.offsets[stacked ? 0 : s];
        let a: number;
        let b: number;
        let end: BarEnd;
        const sign: 1 | -1 = v >= 0 ? 1 : -1;
        if (stack) {
          const seg = stack.segments[xi][s]!;
          a = scale(seg.y0);
          b = scale(seg.y1);
          const first = seg.y0 === 0;
          const last = s === (sign === 1 ? lastUp : lastDown);
          // 2 px of surface between neighbours: one pixel off each touching edge.
          const dir = b >= a ? 1 : -1;
          if (!first) a += dir;
          if (!last) b -= dir;
          end = last ? (rows ? (sign === 1 ? "right" : "left") : sign === 1 ? "top" : "bottom") : "none";
        } else {
          a = scale(0);
          b = scale(v);
          end = rows ? (sign === 1 ? "right" : "left") : sign === 1 ? "top" : "bottom";
        }
        const lo = Math.min(a, b);
        const len = Math.max(1, Math.abs(b - a));
        const rect = rows ? { x: lo, y: catStart + off, w: len, h: geom.bar } : { x: catStart + off, y: lo, w: geom.bar, h: len };
        const tip = rows
          ? { x: sign === 1 ? lo + len : lo, y: rect.y + rect.h / 2 }
          : { x: rect.x + rect.w / 2, y: sign === 1 ? lo : lo + len };
        out.push({ xi, s, value: v, path: barPath(rect.x, rect.y, rect.w, rect.h, end), rect, color, tip, sign });
      });
    }
    return out;
  }, [m, series, matrix, stack, rows, x0, y0, slot, geom, scale, stacked, emphasis, model.highlightX]);

  // ---- hits: bigger than the bars
  const hits: HitBox[] = useMemo(() => {
    return bars.map((b) => {
      const per = slot / (stacked ? 1 : k);
      let box: { x: number; y: number; w: number; h: number };
      if (rows) {
        const h = Math.max(18, per);
        const cy = b.rect.y + b.rect.h / 2;
        box = { x: x0, y: cy - h / 2, w: x1 - x0 + 8, h };
      } else if (stacked) {
        const w = Math.max(24, b.rect.w);
        const h = Math.max(16, b.rect.h);
        box = { x: b.rect.x + b.rect.w / 2 - w / 2, y: b.rect.y + b.rect.h / 2 - h / 2, w, h };
      } else {
        const w = Math.max(14, per);
        const cx = x0 + b.xi * slot + (b.s + 0.5) * per;
        box = { x: cx - w / 2, y: y0, w, h: y1 - y0 };
      }
      return { xi: b.xi, si: b.s, ...box, mark: { ...b.rect, shape: "rect" as const }, ax: b.tip.x, ay: b.tip.y };
    });
  }, [bars, rows, stacked, k, slot, x0, x1, y0, y1]);

  const describe = useCallback(
    (xi: number, s: number) => describePoint(model, panel, xi, series[s].index, { shape: "bar", rows: variant === "bar" ? [series[s]] : series }),
    [model, panel, series, variant],
  );

  // ---- labels on the category axis (columns): wrapped to two lines when they fit that way, else thinned
  const colLabels = useMemo(() => {
    if (rows) return null;
    const widths = xTexts.map((t) => measure(t));
    const positions = xTexts.map((_, i) => x0 + (i + 0.5) * slot);
    if (Math.max(...widths) <= slot - 6 || m > 12) {
      const shown = pickLabels(positions, widths.slice(0, m), 8, model.highlightX);
      return { lines: xTexts.map((t) => [fitLabel(t, Math.max(40, slot * 2 - 8), measure)]), shown };
    }
    const lines = xTexts.map((t) => wrapLabel(t, slot - 6, measure, 2));
    return { lines, shown: positions.map((_, i) => i) };
  }, [rows, xTexts, measure, x0, slot, m, model.highlightX]);

  // ---- values written at tips, without colliding
  const written = useMemo(() => {
    const boxes: Array<{ l: number; r: number; t: number; b: number }> = [];
    const out: Array<{ key: string; x: number; y: number; text: string; anchor: "start" | "middle" | "end" }> = [];
    for (const b of bars) {
      const key = `${b.xi}:${b.s}`;
      if (!wantLabels.has(key)) continue;
      const text = labelTexts[b.xi][b.s];
      const w = strong(text);
      let x: number;
      let y: number;
      let anchor: "start" | "middle" | "end";
      if (rows) {
        anchor = b.sign === 1 ? "start" : "end";
        x = b.tip.x + (b.sign === 1 ? 6 : -6);
        y = b.tip.y + 4;
      } else {
        anchor = "middle";
        x = b.tip.x;
        y = b.sign === 1 ? b.tip.y - 6 : b.tip.y + 15;
      }
      const l = anchor === "middle" ? x - w / 2 : anchor === "start" ? x : x - w;
      const box = { l, r: l + w, t: y - 12, b: y + 3 };
      if (box.l < 0 || box.r > width + 2) continue;
      if (boxes.some((o) => box.l < o.r + 4 && o.l < box.r + 4 && box.t < o.b && o.t < box.b)) continue;
      boxes.push(box);
      out.push({ key, x, y, text, anchor });
    }
    return out;
  }, [bars, wantLabels, labelTexts, rows, strong, width]);

  // ---- values written inside long stacked segments (rows only)
  const inside = useMemo(() => {
    if (!stacked || !rows) return [];
    return bars
      .map((b) => {
        const text = labelTexts[b.xi][b.s];
        const w = strong(text);
        return b.rect.w >= w + 14
          ? { key: `${b.xi}:${b.s}`, x: b.rect.x + b.rect.w / 2, y: b.rect.y + b.rect.h / 2 + 4, text, slot: series[b.s].slot }
          : null;
      })
      .filter((v): v is { key: string; x: number; y: number; text: string; slot: number | null } => v !== null);
  }, [stacked, rows, bars, labelTexts, strong, series]);

  // Sideways bars write the value axis along the bottom: keep the tick labels that fit without touching.
  const shownTicks = new Set(rows ? pickLabels(domain.ticks.map((t) => scale(t)), tickText.map((t) => measure(t)), 10) : domain.ticks.map((_, i) => i));

  const kind = stacked ? labels.chartKind.stacked_bar : k > 1 ? labels.chartKind.grouped_bar : labels.chartKind.bar;

  return (
    <>
      <ChartFrame
        width={width}
        height={height}
        hits={hits}
        nx={m}
        ns={k}
        sideways={rows}
        label={`${visual.title}: ${kind}${caption ? ` (${caption})` : ""}`}
        describedBy={summaryId}
        describe={describe}
      >
        {({ active }) => (
          <g>
            {caption && (
              <text className="cv-tick cv-caption" x={0} y={11}>
                {caption}
              </text>
            )}
            {/* grid and value axis */}
            {domain.ticks.map((t, ti) =>
              rows ? (
                <g key={t}>
                  <line className={t === 0 && domain.min < 0 ? "cv-zero" : "cv-gl"} x1={scale(t)} x2={scale(t)} y1={y0} y2={y1} />
                  {shownTicks.has(ti) && (
                    <text className="cv-tick" x={scale(t)} y={y1 + 17} textAnchor="middle">
                      {tickText[ti]}
                    </text>
                  )}
                </g>
              ) : (
                <g key={t}>
                  <line className={t === 0 && domain.min < 0 ? "cv-zero" : "cv-gl"} x1={x0 - 4} x2={x1 + 8} y1={scale(t)} y2={scale(t)} />
                  <text className="cv-tick" x={x0 - 10} y={scale(t) + 4} textAnchor="end">
                    {tickText[ti]}
                  </text>
                </g>
              ),
            )}
            {rows ? (
              <line className="cv-axis" x1={baseline} x2={baseline} y1={y0} y2={y1} />
            ) : (
              <line className="cv-axis" x1={x0 - 4} x2={x1 + 8} y1={baseline} y2={baseline} />
            )}
            {/* highlighted positions */}
            {[...model.highlightX]
              .filter((xi) => xi < m)
              .map((xi) =>
                rows ? (
                  <rect key={xi} className="cv-band" x={x0 - 6} y={y0 + xi * slot} width={x1 - x0 + 14} height={slot} />
                ) : (
                  <g key={xi}>
                    <rect className="cv-band" x={x0 + xi * slot} y={y0} width={slot} height={y1 - y0} />
                    <path className="cv-flag" d={`M${x0 + (xi + 0.5) * slot - 4} ${y0}h8l-4 6z`} />
                  </g>
                ),
              )}
            {/* bars */}
            {bars.map((b) => {
              const srs = series[b.s];
              const isActive = active?.xi === b.xi && active.si === b.s;
              const dim = dimOthers && !model.highlightSeries.has(srs.key);
              return (
                <path
                  key={`${b.xi}:${b.s}`}
                  className="cv-bar"
                  d={b.path}
                  fill={b.color}
                  data-active={isActive || undefined}
                  opacity={dim ? 0.38 : 1}
                />
              );
            })}
            {inside.map((v) => (
              <text key={v.key} className="cv-inside" data-slot={v.slot === null ? undefined : v.slot + 1} x={v.x} y={v.y} textAnchor="middle">
                {v.text}
              </text>
            ))}
            {written.map((w) => (
              <text key={w.key} className="cv-value" x={w.x} y={w.y} textAnchor={w.anchor}>
                {w.text}
              </text>
            ))}
            {/* missing values: a small hollow mark on the baseline, so a gap reads as "not reported" */}
            {matrix.flatMap((row, xi) =>
              row.map((v, s) => {
                if (v !== null) return null;
                const per = slot / (stacked ? 1 : k);
                const c = rows ? { x: baseline + 7, y: y0 + xi * slot + (s + 0.5) * per } : { x: x0 + xi * slot + (s + 0.5) * per, y: baseline - 7 };
                return <circle key={`n${xi}:${s}`} className="cv-null" cx={c.x} cy={c.y} r={3.5} />;
              }),
            )}
            {/* category labels */}
            {rows
              ? catLabels.map((t, xi) => (
                  <text
                    key={xi}
                    className={model.highlightX.has(xi) ? "cv-tick cv-tick-strong" : "cv-tick"}
                    x={x0 - 10}
                    y={y0 + xi * slot + slot / 2 + 4}
                    textAnchor="end"
                  >
                    {t}
                  </text>
                ))
              : colLabels?.shown.map((i) => (
                  <text
                    key={i}
                    className={model.highlightX.has(i) ? "cv-tick cv-tick-strong" : "cv-tick"}
                    x={x0 + (i + 0.5) * slot}
                    y={y1 + 18}
                    textAnchor="middle"
                  >
                    {colLabels.lines[i].map((line, li) => (
                      <tspan key={li} x={x0 + (i + 0.5) * slot} dy={li === 0 ? 0 : 14}>
                        {line}
                      </tspan>
                    ))}
                  </text>
                ))}
          </g>
        )}
      </ChartFrame>
      {folded && (
        <p className="cv-fold">
          <span>{labels.showing(m, nAll)}</span>
          <button type="button" className="link-btn" aria-expanded={expanded} onClick={() => setExpanded((e) => !e)}>
            {expanded ? labels.showFewer : labels.showAll(nAll)}
          </button>
        </p>
      )}
    </>
  );
}
