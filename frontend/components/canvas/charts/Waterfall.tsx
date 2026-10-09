"use client";

/**
 * Waterfall (a bridge, e.g. revenue → profit): a start bar and an end bar on the baseline with increases and
 * decreases floating between them. The first and last rows are the totals unless a row says otherwise (`kind`). Three
 * encodings carry the direction, so colour is never the only one: the bar's position (rising or falling from the
 * running level), the signed value written at its tip, and the legend's words. The running level is layout only; every
 * number shown is the row's own value from the document.
 */

import { useCallback, useMemo } from "react";

import { formatDelta, formatTick, formatValue, MINUS, unitCaption } from "@/lib/canvas/format";
import type { Model } from "@/lib/canvas/model";
import { barGeometry, barPath, fitLabel, linearScale, niceTicks, pickLabels, waterfallLayout, wrapLabel, type BarEnd, type WaterfallKind } from "@/lib/canvas/scales";
import type { Unit, Visual } from "@/lib/canvas/types";

import { ChartFrame, type Described, type HitBox } from "../ChartFrame";
import { ChartBox, Legend, xText, type LegendItem } from "../chart-common";
import { useMeasure } from "../measure";
import { usePanel } from "../panel-context";

/** Which rows are totals (standing on the baseline): the ones that say so, else the first and the last. */
export function waterfallTotals(visual: Visual): boolean[] {
  const rows = visual.rows;
  return rows.map((r, i) => (r.kind ? r.kind === "total" : i === 0 || i === rows.length - 1));
}

export function Waterfall({ model, summaryId }: { model: Model; summaryId: string }) {
  const { labels } = usePanel();
  const kinds = useMemo(() => waterfallTotals(model.visual), [model]);
  const items: LegendItem[] = [
    { key: "total", label: labels.total, color: "var(--viz-total)", shape: "bar" },
    { key: "up", label: `▲ ${labels.increase}`, color: "var(--viz-pos)", shape: "bar" },
    { key: "down", label: `▼ ${labels.decrease}`, color: "var(--viz-neg)", shape: "bar" },
  ];
  return (
    <>
      <Legend items={items} label={labels.series} />
      <ChartBox>{(width) => <WaterfallInner model={model} totals={kinds} width={width} summaryId={summaryId} />}</ChartBox>
    </>
  );
}

const COLOR: Record<WaterfallKind, string> = {
  total: "var(--viz-total)",
  up: "var(--viz-pos)",
  down: "var(--viz-neg)",
  none: "transparent",
};

/** A figure for a bar's tip: the number alone (the axis caption carries the unit), signed for increases and decreases. */
function tipText(value: number, unit: Unit | null, signed: boolean): string {
  const sign = value < 0 ? MINUS : signed ? "+" : "";
  const num = formatTick(Math.abs(value), unit);
  return `${sign}${num}`;
}

function WaterfallInner({ model, totals, width, summaryId }: { model: Model; totals: boolean[]; width: number; summaryId: string }) {
  const panel = usePanel();
  const { visual, labels } = panel;
  const lang = visual.language;
  const measure = useMeasure(12);
  const strong = useMeasure(12, 600);
  const series = model.series[0];
  const unit = series?.unit ?? visual.unit;
  const n = model.xs.length;
  const key = series?.key ?? "";

  const layout = useMemo(
    () => waterfallLayout(model.xs.map((_, xi) => ({ value: model.value(xi, key), total: totals[xi] }))),
    [model, totals, key],
  );
  const domain = useMemo(() => niceTicks(layout.min, layout.max, width < 480 ? 3 : 4), [layout, width]);
  const rows = width / Math.max(1, n) < 64;
  const texts = useMemo(() => model.xs.map((x) => xText(visual, x)), [model, visual]);
  const tickText = domain.ticks.map((t) => formatTick(t, unit));
  const caption = unitCaption(unit, lang);

  const tickW = Math.max(24, ...tickText.map((t) => measure(t)));
  const labelFont = (xi: number) => (model.highlightX.has(xi) ? strong : measure);
  const labelW = Math.min(width * 0.4, Math.max(48, ...texts.map((t, xi) => labelFont(xi)(t))) + 4);
  const CAP = 28;
  const plotH = width < 480 ? 160 : 200;
  const rowSlot = 28;
  const x0 = rows ? labelW + 12 : tickW + 14;
  const x1 = width - (rows ? 14 + 64 : 14);
  const y0 = CAP;
  const y1 = y0 + (rows ? n * rowSlot : plotH);
  const axisH = rows ? 26 : 40;
  const height = y1 + axisH;
  const slot = rows ? rowSlot : (x1 - x0) / Math.max(1, n);
  const geom = barGeometry(slot, 1, { maxBar: rows ? 20 : 28 });
  const scale = rows ? linearScale(domain.min, domain.max, x0, x1) : linearScale(domain.min, domain.max, y1, y0);

  const shownTicks = new Set(rows ? pickLabels(domain.ticks.map((t) => scale(t)), tickText.map((t) => measure(t)), 10) : domain.ticks.map((_, i) => i));

  const bars = useMemo(
    () =>
      layout.bars.map((b, xi) => {
        if (b.kind === "none") return null;
        const a = scale(b.from);
        const c = scale(b.to);
        const lo = Math.min(a, c);
        const len = Math.max(1, Math.abs(c - a));
        const start = (rows ? y0 : x0) + xi * slot + geom.offsets[0];
        const rect = rows ? { x: lo, y: start, w: len, h: geom.bar } : { x: start, y: lo, w: geom.bar, h: len };
        const rising = b.to >= b.from;
        const end: BarEnd = rows ? (rising ? "right" : "left") : rising ? "top" : "bottom";
        return { xi, rect, kind: b.kind, path: barPath(rect.x, rect.y, rect.w, rect.h, end), level: scale(b.level), rising };
      }),
    [layout, scale, rows, y0, x0, slot, geom],
  );

  const hits: HitBox[] = useMemo(() => {
    const out: HitBox[] = [];
    bars.forEach((b) => {
      if (!b) return;
      const box = rows
        ? { x: x0 - 4, y: y0 + b.xi * slot, w: x1 - x0 + 70, h: slot }
        : { x: x0 + b.xi * slot, y: y0, w: slot, h: y1 - y0 };
      const tip = rows ? { x: b.rect.x + b.rect.w, y: b.rect.y + b.rect.h / 2 } : { x: b.rect.x + b.rect.w / 2, y: b.rect.y };
      out.push({ xi: b.xi, si: 0, ...box, mark: { ...b.rect, shape: "rect" }, ax: tip.x, ay: tip.y });
    });
    return out;
  }, [bars, rows, slot, x0, x1, y0, y1]);

  const describe = useCallback(
    (xi: number): Described => {
      const v = model.value(xi, key);
      const kind = layout.bars[xi]?.kind ?? "none";
      const word = kind === "total" ? labels.total : kind === "up" ? labels.increase : labels.decrease;
      const value = kind === "total" ? formatValue(v, unit, lang) : formatDelta(v ?? 0, "abs", unit, lang);
      const spoken = kind === "total" ? formatValue(v, unit, lang, { spoken: true }) : formatDelta(v ?? 0, "abs", unit, lang, { spoken: true });
      const cell = model.cell(xi, key);
      const src = cell ? panel.resolve(cell) : null;
      const x = texts[xi];
      return {
        aria: [`${x}, ${word}: ${spoken}`, src ? `${labels.source}: ${src.filename}${src.page !== null ? `, ${labels.page} ${src.page}` : ""}` : null].filter(Boolean).join(". "),
        tip: {
          title: x,
          rows: [{ color: COLOR[kind], shape: "bar", label: word, value, calculated: series?.calculated ?? false, active: true }],
          source: src ? { filename: src.filename, page: src.page, cellText: src.cellText } : null,
          formula: null,
        },
        cell,
      };
    },
    [model, key, layout, labels, unit, lang, texts, panel, series],
  );

  // Every bar's own value at its tip (a bridge has few steps); a label that would collide with a neighbour is left to the tooltip.
  const written = useMemo(() => {
    const out: Array<{ xi: number; x: number; y: number; text: string; anchor: "start" | "middle" }> = [];
    const taken: Array<{ l: number; r: number }> = [];
    const order = [...bars.keys()].sort((a, b) => (layout.bars[b]?.kind === "total" ? 1 : 0) - (layout.bars[a]?.kind === "total" ? 1 : 0));
    for (const xi of order) {
      const b = bars[xi];
      if (!b) continue;
      const v = layout.bars[xi].value;
      if (v === null) continue;
      const text = tipText(v, unit, layout.bars[xi].kind !== "total");
      const w = strong(text);
      if (rows) {
        out.push({ xi, x: b.rect.x + b.rect.w + 6, y: b.rect.y + b.rect.h / 2 + 4, text, anchor: "start" });
        continue;
      }
      const cx = b.rect.x + b.rect.w / 2;
      const box = { l: cx - w / 2, r: cx + w / 2 };
      if (taken.some((t) => box.l < t.r + 4 && t.l < box.r + 4)) continue;
      taken.push(box);
      const below = !b.rising && b.kind !== "total";
      out.push({ xi, x: cx, y: below ? b.rect.y + b.rect.h + 15 : b.rect.y - 6, text, anchor: "middle" });
    }
    return out;
  }, [bars, layout, unit, rows, strong]);

  const wrapped = useMemo(
    () => (rows ? null : texts.map((t) => wrapLabel(t, slot - 2, measure, 2))),
    [rows, texts, slot, measure],
  );

  return (
    <ChartFrame
      width={width}
      height={height}
      hits={hits}
      nx={n}
      ns={1}
      sideways={rows}
      label={`${visual.title}: ${labels.chartKind.waterfall}`}
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
          {[...model.highlightX].map((xi) =>
            rows ? (
              <rect key={xi} className="cv-band" x={x0 - 6} y={y0 + xi * slot} width={x1 - x0 + 76} height={slot} />
            ) : (
              <g key={xi}>
                <rect className="cv-band" x={x0 + xi * slot} y={y0} width={slot} height={y1 - y0} />
                <path className="cv-flag" d={`M${x0 + (xi + 0.5) * slot - 4} ${y0}h8l-4 6z`} />
              </g>
            ),
          )}
          {/* the running level joining one bar to the next */}
          {bars.map((b, i) => {
            const next = bars.slice(i + 1).find((x) => x !== null);
            if (!b || !next) return null;
            return rows ? (
              <line key={`c${i}`} className="cv-connector" x1={b.level} x2={b.level} y1={b.rect.y + b.rect.h} y2={next.rect.y} />
            ) : (
              <line key={`c${i}`} className="cv-connector" x1={b.rect.x + b.rect.w} x2={next.rect.x} y1={b.level} y2={b.level} />
            );
          })}
          {bars.map((b) =>
            b ? <path key={b.xi} className="cv-bar" d={b.path} fill={COLOR[b.kind]} data-active={(active?.xi === b.xi) || undefined} /> : null,
          )}
          {written.map((w) => (
            <text key={w.xi} className="cv-value" x={w.x} y={w.y} textAnchor={w.anchor}>
              {w.text}
            </text>
          ))}
          {rows
            ? texts.map((t, xi) => (
                <text key={xi} className={model.highlightX.has(xi) ? "cv-tick cv-tick-strong" : "cv-tick"} x={x0 - 10} y={y0 + xi * slot + slot / 2 + 4} textAnchor="end">
                  {fitLabel(t, labelW, labelFont(xi))}
                </text>
              ))
            : wrapped?.map((lines, xi) => (
                <text key={xi} className={model.highlightX.has(xi) ? "cv-tick cv-tick-strong" : "cv-tick"} x={x0 + (xi + 0.5) * slot} y={y1 + 18} textAnchor="middle">
                  {lines.map((line, li) => (
                    <tspan key={li} x={x0 + (xi + 0.5) * slot} dy={li === 0 ? 0 : 14}>
                      {line}
                    </tspan>
                  ))}
                </text>
              ))}
        </g>
      )}
    </ChartFrame>
  );
}
