"use client";

/**
 * Exact figures as a table: the `table` visual itself, and the "View as table" twin of every other kind (the
 * accessibility equivalent of a chart). Numbers align to the right in tabular figures; each one that came from a cell
 * opens its source; calculated columns say so in their header; highlighted rows are marked in words as well as style.
 */

import { formatCell, formatDate, formatDelta, formatValue, isIsoDate, unitCaption } from "@/lib/canvas/format";
import type { Model } from "@/lib/canvas/model";
import type { Visual } from "@/lib/canvas/types";

import { xText } from "../chart-common";
import { CalcBadge, CellButton, usePanel } from "../panel-context";

/** The rows-and-series table of a chart or of a `table` visual. */
export function DataTable({ model, extra }: { model: Model; extra?: { header: string; values: string[] } }) {
  const { visual, labels } = usePanel();
  const lang = visual.language;
  return (
    <div className="cv-table-wrap" role="region" aria-label={`${visual.title}: ${labels.chartKind.table}`} tabIndex={0}>
      <table className="cv-table">
        <thead>
          <tr>
            <th scope="col">{visual.x?.label ?? ""}</th>
            {extra && <th scope="col">{extra.header}</th>}
            {model.series.map((s) => (
              <th key={s.key} scope="col" className="num">
                <span className="cv-th-name">{s.label}</span>
                {unitCaption(s.unit, lang) && <span className="cv-th-unit">{unitCaption(s.unit, lang)}</span>}
                {s.calculated && <CalcBadge />}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {model.xs.map((x, xi) => {
            const strong = model.highlightX.has(xi);
            return (
              <tr key={xi} data-highlight={strong || undefined}>
                <th scope="row">
                  {xText(visual, x)}
                  {strong && <span className="visually-hidden">, {labels.highlighted}</span>}
                </th>
                {extra && <td>{extra.values[xi]}</td>}
                {model.series.map((s) => {
                  const v = model.value(xi, s.key);
                  return (
                    <td key={s.key} className="num" data-strong={model.highlightSeries.has(s.key) || undefined}>
                      {v === null ? (
                        <>
                          <span aria-hidden="true">—</span>
                          <span className="visually-hidden">{labels.noValue}</span>
                        </>
                      ) : (
                        <CellButton cell={model.cell(xi, s.key)}>{formatCell(v, s.unit)}</CellButton>
                      )}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** Key figures and comparisons as a table: metric, value, change (marked calculated), source. */
export function TilesTable({ visual }: { visual: Visual }) {
  const { labels } = usePanel();
  const lang = visual.language;
  return (
    <div className="cv-table-wrap" role="region" aria-label={`${visual.title}: ${labels.chartKind.table}`} tabIndex={0}>
      <table className="cv-table">
        <thead>
          <tr>
            <th scope="col" />
            <th scope="col" className="num">
              {visual.unit ? unitCaption(visual.unit, lang) || "" : ""}
            </th>
            <th scope="col" className="num">
              Δ
            </th>
          </tr>
        </thead>
        <tbody>
          {visual.tiles.map((t, i) => (
            <tr key={i}>
              <th scope="row">{t.label}</th>
              <td className="num">
                <CellButton cell={t.cell}>{formatValue(t.value, t.unit ?? visual.unit, lang)}</CellButton>
              </td>
              <td className="num">
                {t.delta ? (
                  <>
                    {formatDelta(t.delta.value, t.delta.kind, t.unit ?? visual.unit, lang)} <CalcBadge />
                  </>
                ) : (
                  <span aria-hidden="true">—</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Dates and obligations as a table: date, what, detail, source. */
export function EventsTable({ visual }: { visual: Visual }) {
  const { labels, resolve } = usePanel();
  const lang = visual.language;
  return (
    <div className="cv-table-wrap" role="region" aria-label={`${visual.title}: ${labels.chartKind.table}`} tabIndex={0}>
      <table className="cv-table">
        <thead>
          <tr>
            <th scope="col">{visual.x?.label || ""}</th>
            <th scope="col" />
            <th scope="col">{labels.source}</th>
          </tr>
        </thead>
        <tbody>
          {visual.events.map((e, i) => (
            <tr key={i}>
              <th scope="row">{isIsoDate(e.date) ? formatDate(e.date, lang) : e.date}</th>
              <td>
                {e.label}
                {e.detail && <span className="cv-detail">{e.detail}</span>}
              </td>
              <td>
                <CellButton cell={e.cell}>{e.cell ? `${resolve(e.cell).filename}${e.cell.page !== null ? ` · ${e.cell.page}` : ""}` : ""}</CellButton>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
