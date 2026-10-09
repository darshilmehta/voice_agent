"use client";

/**
 * Line chart (trend over time). 2 px lines, 10 px markers with a 2 px surface ring, hairline solid grid, a legend for
 * two or more series, direct value labels on the line ends only, a crosshair that snaps to the nearest position, and
 * the highlight as a wash behind its positions plus a heavier line. Series in different units never share an axis:
 * each unit gets its own small chart, stacked, sharing the x axis and one crosshair.
 */

import { useCallback, useMemo } from "react";

import { formatTick, formatValue, isoToTime, unitCaption } from "@/lib/canvas/format";
import { seriesColor, type Model } from "@/lib/canvas/model";
import {
  areaPath,
  fitLabel,
  linePath,
  linearScale,
  niceDomain,
  pickLabels,
  proportionalPositions,
  type Pt,
} from "@/lib/canvas/scales";

import { ChartFrame, type HitBox } from "../ChartFrame";
import { ChartBox, Legend, describePoint, seriesLegend, xText } from "../chart-common";
import { useMeasure } from "../measure";
import { usePanel } from "../panel-context";

export function LineChart({ model, summaryId }: { model: Model; summaryId: string }) {
  return <ChartBox>{(width) => <LineInner model={model} width={width} summaryId={summaryId} />}</ChartBox>;
}

const CAPTION_H = 26;
const GROUP_GAP = 14;
const AXIS_H = 32;
const SPOT = 28;

function LineInner({ model, width, summaryId }: { model: Model; width: number; summaryId: string }) {
  const panel = usePanel();
  const { visual, labels } = panel;
  const lang = visual.language;
  const measure = useMeasure(12);
  const measureStrong = useMeasure(12, 600);
  const narrow = width < 480;
  // Several units stack several small charts: each is a little shorter so the whole fits a screen.
  const plotH = model.groups.length > 1 ? (narrow ? 118 : 142) : narrow ? 150 : 188;
  const n = model.xs.length;

  const layout = useMemo(() => {
    const domains = model.groups.map((g) =>
      niceDomain(
        g.series.flatMap((s) => model.xs.map((_, xi) => model.value(xi, s.key))),
        { zero: "auto", count: narrow ? 3 : 4 },
      ),
    );
    const tickText = domains.map((d, gi) => d.ticks.map((t) => formatTick(t, model.groups[gi].unit)));
    const left = Math.max(28, ...tickText.flat().map((t) => measure(t))) + 12;
    const x0 = left;
    const x1 = width - 14;
    const times = model.xs.map((x) => isoToTime(x));
    const prop = visual.x?.type === "date" ? proportionalPositions(times, x0 + 10, x1 - 10) : null;
    const step = (x1 - x0) / Math.max(1, n);
    const xs = model.xs.map((_, i) => (prop ? prop[i] : x0 + step * (i + 0.5)));
    let minStep = Infinity;
    for (let i = 1; i < xs.length; i++) minStep = Math.min(minStep, Math.abs(xs[i] - xs[i - 1]));
    if (!Number.isFinite(minStep)) minStep = x1 - x0;
    return { domains, tickText, left, x0, x1, xs, minStep, byTime: prop !== null };
  }, [model, width, narrow, n, measure, visual.x?.type]);

  const { domains, tickText, left, x0, x1, xs, minStep, byTime } = layout;
  const groupTop = useCallback((gi: number) => gi * (CAPTION_H + plotH + GROUP_GAP), [plotH]);
  const yTop = useCallback((gi: number) => groupTop(gi) + CAPTION_H, [groupTop]);
  const height = model.groups.length * (CAPTION_H + plotH) + (model.groups.length - 1) * GROUP_GAP + AXIS_H;
  const ySc = useMemo(() => domains.map((d, gi) => linearScale(d.min, d.max, yTop(gi) + plotH, yTop(gi))), [domains, plotH, yTop]);

  // Which series sits in which group, and each series' point positions.
  const seriesGroup = useMemo(() => {
    const m = new Map<string, number>();
    model.groups.forEach((g, gi) => g.series.forEach((s) => m.set(s.key, gi)));
    return m;
  }, [model]);

  const point = useCallback(
    (xi: number, si: number): Pt | null => {
      const s = model.series[si];
      const gi = seriesGroup.get(s.key) ?? 0;
      const v = model.value(xi, s.key);
      return v === null ? null : { x: xs[xi], y: ySc[gi](v) };
    },
    [model, seriesGroup, xs, ySc],
  );

  const hits: HitBox[] = useMemo(() => {
    const out: HitBox[] = [];
    const hw = Math.min(SPOT, Math.max(14, minStep));
    model.series.forEach((s, si) => {
      const gi = seriesGroup.get(s.key) ?? 0;
      for (let xi = 0; xi < n; xi++) {
        const p = point(xi, si) ?? { x: xs[xi], y: yTop(gi) + plotH };
        out.push({
          xi,
          si,
          x: p.x - hw / 2,
          y: p.y - SPOT / 2,
          w: hw,
          h: SPOT,
          mark: { x: p.x - 5, y: p.y - 5, w: 10, h: 10, shape: "circle" },
          ax: p.x,
          ay: p.y,
        });
      }
    });
    return out;
  }, [model, seriesGroup, n, point, xs, minStep, plotH, yTop]);

  const describe = useCallback((xi: number, si: number) => describePoint(model, panel, xi, si, { shape: "line" }), [model, panel]);

  // X labels: all that fit, ending on the last; highlighted ones kept when they fit.
  const xLabels = useMemo(() => {
    // Dates placed by time have no regular step to size a label by: they may be as wide as a few fit across the chart.
    const cap = byTime ? Math.max(70, Math.min(150, (x1 - x0) / Math.min(n, 4))) : Math.max(44, Math.min(150, minStep * (n <= 12 ? 1 : 2) - 8));
    const text = model.xs.map((x) => fitLabel(xText(visual, x), cap, measure));
    const widths = text.map((t, i) => (model.highlightX.has(i) ? measureStrong(t) : measure(t)));
    const shown = pickLabels(xs, widths, 10, model.highlightX);
    return { text, shown };
  }, [model, visual, xs, minStep, n, measure, measureStrong, byTime, x0, x1]);

  // End labels: the value at the last point of each series, kept only where it doesn't collide with another.
  const endLabels = useMemo(() => {
    const out: Array<{ si: number; x: number; y: number; text: string; w: number }> = [];
    model.series.forEach((s, si) => {
      let li = -1;
      for (let xi = n - 1; xi >= 0; xi--) {
        if (model.value(xi, s.key) !== null) {
          li = xi;
          break;
        }
      }
      if (li < 0) return;
      const p = point(li, si);
      if (!p) return;
      const text = formatValue(model.value(li, s.key), s.unit, lang);
      const w = measureStrong(text);
      const x = Math.min(width - 4, p.x + 2);
      const y = p.y - 11;
      const clash = out.some((o) => Math.abs(o.y - y) < 15 && x - w < o.x && o.x - o.w < x);
      if (!clash) out.push({ si, x, y: Math.max(12, y), text, w });
    });
    return out;
  }, [model, n, point, lang, width, measureStrong]);

  const legend = seriesLegend(model, "line");
  const dimOthers = model.highlightSeries.size > 0;

  return (
    <>
      <Legend items={legend} label={labels.series} />
      <ChartFrame
        width={width}
        height={height}
        hits={hits}
        nx={n}
        ns={model.series.length}
        label={`${visual.title}: ${labels.chartKind.line}`}
        describedBy={summaryId}
        describe={describe}
        snap={{ xs, x0, x1 }}
        overlay={(active) =>
          active ? (
            <line
              className="cv-crosshair"
              x1={xs[active.xi]}
              x2={xs[active.xi]}
              y1={yTop(0)}
              y2={yTop(model.groups.length - 1) + plotH}
            />
          ) : null
        }
      >
        {({ active }) => (
          <g>
            {model.groups.map((g, gi) => {
              const y0 = yTop(gi);
              const y1 = y0 + plotH;
              const d = domains[gi];
              const sc = ySc[gi];
              const single = g.series.length === 1;
              return (
                <g key={gi}>
                  <text className="cv-tick cv-caption" x={0} y={groupTop(gi) + 11}>
                    {unitCaption(g.unit, lang)}
                  </text>
                  {d.ticks.map((t, ti) => (
                    <g key={t}>
                      <line className={t === 0 && d.min < 0 ? "cv-zero" : "cv-gl"} x1={x0 - 4} x2={x1 + 8} y1={sc(t)} y2={sc(t)} />
                      <text className="cv-tick" x={left - 8} y={sc(t) + 4} textAnchor="end">
                        {tickText[gi][ti]}
                      </text>
                    </g>
                  ))}
                  <line className="cv-axis" x1={x0 - 4} x2={x1 + 8} y1={y1} y2={y1} />
                  {[...model.highlightX].map((xi) => (
                    <g key={`h${xi}`}>
                      <rect className="cv-band" x={xs[xi] - Math.min(minStep, 60) / 2} y={y0} width={Math.min(minStep, 60)} height={plotH} />
                      <path className="cv-flag" d={`M${xs[xi] - 4} ${y0}h8l-4 6z`} />
                    </g>
                  ))}
                  {g.series.map((s) => {
                    const si = s.index;
                    const pts: Array<Pt | null> = model.xs.map((_, xi) => point(xi, si));
                    const color = seriesColor(s.slot);
                    const strong = model.highlightSeries.has(s.key);
                    const dim = dimOthers && !strong;
                    return (
                      <g key={s.key} opacity={dim ? 0.38 : 1}>
                        {single && <path className="cv-area" d={areaPath(pts, y1)} fill={color} />}
                        <path className="cv-line" d={linePath(pts)} stroke={color} strokeWidth={strong ? 3 : 2} />
                      </g>
                    );
                  })}
                  {g.series.map((s) => {
                    const si = s.index;
                    const color = seriesColor(s.slot);
                    const dim = dimOthers && !model.highlightSeries.has(s.key);
                    return (
                      <g key={`m${s.key}`} opacity={dim ? 0.38 : 1}>
                        {model.xs.map((_, xi) => {
                          const p = point(xi, si);
                          const isActive = active?.xi === xi && active.si === si;
                          if (!p) {
                            return <circle key={xi} className="cv-null" cx={xs[xi]} cy={y1} r={3.5} />;
                          }
                          const important = model.highlightX.has(xi) || xi === n - 1 || isActive;
                          if (n > 12 && !important) return null;
                          return (
                            <circle
                              key={xi}
                              className="cv-dot"
                              cx={p.x}
                              cy={p.y}
                              r={isActive ? 6.5 : model.highlightX.has(xi) ? 6 : 5}
                              fill={color}
                            />
                          );
                        })}
                      </g>
                    );
                  })}
                  {single &&
                    [...model.highlightX].map((xi) => {
                      const s = g.series[0];
                      const p = point(xi, s.index);
                      if (!p) return null;
                      return (
                        <text key={`v${xi}`} className="cv-value" x={p.x} y={p.y - 11} textAnchor="middle">
                          {formatValue(model.value(xi, s.key), s.unit, lang)}
                        </text>
                      );
                    })}
                </g>
              );
            })}
            {endLabels.map((l) => {
              const s = model.series[l.si];
              const gi = seriesGroup.get(s.key) ?? 0;
              // A highlighted position already carries its own value label on a single-series chart.
              if (model.groups[gi].series.length === 1 && model.highlightX.has(n - 1)) return null;
              return (
                <text key={l.si} className="cv-value" x={l.x} y={l.y} textAnchor="end">
                  {l.text}
                </text>
              );
            })}
            {xLabels.shown.map((i) => (
              <text
                key={i}
                className={model.highlightX.has(i) ? "cv-tick cv-tick-strong" : "cv-tick"}
                x={xs[i]}
                y={height - AXIS_H + 19}
                textAnchor={xs[i] - x0 < 20 && i === 0 && xLabels.shown.length > 1 ? "start" : i === n - 1 && x1 - xs[i] < 20 ? "end" : "middle"}
              >
                {xLabels.text[i]}
              </text>
            ))}
          </g>
        )}
      </ChartFrame>
    </>
  );
}
