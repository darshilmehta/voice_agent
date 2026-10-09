"use client";

/**
 * Donut (share of a whole; sparingly). One ring of at most eight parts, each in its fixed palette slot with a 2 px
 * surface gap, and a legend that lists every part with its own figure from the document. The angles are layout only:
 * no percentage is computed here. If the backend calculated a share, it is shown and marked "calculated". A ring with
 * more than eight parts, or fewer than two, is not a donut's job, so it is drawn as bars instead.
 */

import { useCallback, useMemo } from "react";

import { formatValue, unitCaption } from "@/lib/canvas/format";
import { SERIES_SLOTS, seriesColor, type Model } from "@/lib/canvas/model";
import { donutSlices, polar } from "@/lib/canvas/scales";
import type { Calculation } from "@/lib/canvas/types";

import { ChartFrame, type Described, type HitBox } from "../ChartFrame";
import { ChartBox, xText } from "../chart-common";
import { CalcBadge, CellButton, usePanel } from "../panel-context";
import { BarChart } from "./BarChart";

/** A part's colour follows its position, in the palette's fixed order. */
const colorOf = (xi: number) => seriesColor(xi < SERIES_SLOTS ? xi : null);

const PCT = { kind: "percent", currency: null, scale: null, label: "%" } as const;

export function Donut({ model, summaryId }: { model: Model; summaryId: string }) {
  const { labels } = usePanel();
  const key = model.series[0]?.key ?? "";
  const positives = model.xs.filter((_, xi) => (model.value(xi, key) ?? 0) > 0).length;
  if (positives > SERIES_SLOTS || positives < 2) {
    return (
      <>
        {positives > SERIES_SLOTS && <p className="cv-note">{labels.donutTooMany}</p>}
        <BarChart model={model} variant="bar" summaryId={summaryId} />
      </>
    );
  }
  return <ChartBox>{(width) => <DonutInner model={model} width={width} summaryId={summaryId} />}</ChartBox>;
}

function shareOf(calcs: Calculation[], x: string): Calculation | null {
  const n = (s: string) => s.trim().toLowerCase();
  return calcs.find((c) => c.op === "share" && n(c.label).includes(n(x))) ?? null;
}

function DonutInner({ model, width, summaryId }: { model: Model; width: number; summaryId: string }) {
  const panel = usePanel();
  const { visual, labels } = panel;
  const lang = visual.language;
  const series = model.series[0];
  const unit = series.unit;
  const side = width >= 520;
  const size = side ? 208 : Math.min(220, width - 16);
  const R = size / 2 - 6;
  const r0 = R * 0.62;
  const cx = side ? size / 2 : width / 2;
  const cy = size / 2;
  const n = model.xs.length;
  const values = useMemo(() => model.xs.map((_, xi) => model.value(xi, series.key)), [model, series]);
  const slices = useMemo(() => donutSlices(values, cx, cy, r0, R, 2), [values, cx, cy, r0, R]);
  const texts = useMemo(() => model.xs.map((x) => xText(visual, x)), [model, visual]);

  const hits: HitBox[] = useMemo(
    () =>
      slices.map((s) => {
        const p = polar(cx, cy, (r0 + R) / 2, s.mid);
        return {
          xi: s.index,
          si: 0,
          x: p.x - 16,
          y: p.y - 16,
          w: 32,
          h: 32,
          mark: { x: p.x - 6, y: p.y - 6, w: 12, h: 12, shape: "circle" as const },
          ax: polar(cx, cy, R, s.mid).x,
          ay: polar(cx, cy, R, s.mid).y,
        };
      }),
    [slices, cx, cy, r0, R],
  );

  const describe = useCallback(
    (xi: number): Described => {
      const v = values[xi];
      const cell = model.cell(xi, series.key);
      const src = cell ? panel.resolve(cell) : null;
      const share = shareOf(visual.calculations, model.xs[xi]);
      const spoken = formatValue(v, unit, lang, { spoken: true });
      return {
        aria: [
          `${texts[xi]}: ${spoken}`,
          share ? `${formatValue(share.value, share.unit ?? PCT, lang, { spoken: true })}, ${labels.calculated}` : null,
          src ? `${labels.source}: ${src.filename}${src.page !== null ? `, ${labels.page} ${src.page}` : ""}` : null,
        ]
          .filter(Boolean)
          .join(". "),
        tip: {
          title: series.label,
          rows: [
            { color: colorOf(xi), shape: "dot", label: texts[xi], value: formatValue(v, unit, lang), calculated: false, active: true },
            ...(share
              ? [{ color: colorOf(xi), shape: "dot" as const, label: share.label, value: formatValue(share.value, share.unit ?? PCT, lang), calculated: true, active: false }]
              : []),
          ],
          source: src ? { filename: src.filename, page: src.page, cellText: src.cellText } : null,
          formula: share?.formula_text || null,
        },
        cell,
      };
    },
    [values, model, series, panel, visual, unit, lang, labels, texts],
  );

  return (
    <div className="cv-donut" data-side={side || undefined}>
      <ChartFrame
        width={side ? size : width}
        height={size}
        hits={hits}
        nx={n}
        ns={1}
        label={`${visual.title}: ${labels.chartKind.donut}`}
        describedBy={summaryId}
        describe={describe}
      >
        {({ active, bind }) => (
          <g>
            {slices.map((s) => {
              const out = model.highlightX.has(s.index) ? 4 : 0;
              const dx = Math.sin(s.mid) * out;
              const dy = -Math.cos(s.mid) * out;
              return (
                <path
                  key={s.index}
                  className="cv-bar cv-slice"
                  d={s.d}
                  fill={colorOf(s.index)}
                  transform={`translate(${dx} ${dy})`}
                  data-active={(active?.xi === s.index) || undefined}
                  {...bind(s.index, 0)}
                />
              );
            })}
            <text className="cv-center" x={cx} y={cy - 2} textAnchor="middle">
              {unitCaption(unit, lang)}
            </text>
            <text className="cv-center-sub" x={cx} y={cy + 15} textAnchor="middle">
              {series.label}
            </text>
          </g>
        )}
      </ChartFrame>
      <ul className="cv-donut-legend" aria-label={series.label}>
        {model.xs.map((_, xi) => {
          const v = values[xi];
          const share = shareOf(visual.calculations, model.xs[xi]);
          return (
            <li key={xi} data-strong={model.highlightX.has(xi) || undefined}>
              <span className="cv-key" data-shape="bar" style={{ background: colorOf(xi) }} aria-hidden="true" />
              <span className="cv-donut-name">{texts[xi]}</span>
              <CellButton cell={model.cell(xi, series.key)} className="cv-donut-val">
                {formatValue(v, unit, lang)}
              </CellButton>
              {share && (
                <span className="cv-donut-share">
                  {formatValue(share.value, share.unit ?? PCT, lang)} <CalcBadge />
                </span>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
