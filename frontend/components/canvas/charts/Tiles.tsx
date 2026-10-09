"use client";

/**
 * Key-figure tiles and the A-vs-B comparison card. A tile is a number, not a chart: the value leads (proportional
 * figures, the unit small beside it), the figure opens its source, and a change is marked "calculated" with its formula
 * and the cells it was calculated from. A change is never coloured good or bad (the backend doesn't say which way is
 * good): it carries an arrow, a sign and words, and sits in the text colour.
 */

import type { CSSProperties } from "react";

import { directionOf, formatDelta, formatParts, formatValue, type ValueParts } from "@/lib/canvas/format";
import type { CanvasLanguage, Delta as DeltaSpec, Tile, Unit, Visual } from "@/lib/canvas/types";

import { CalcBadge, CellButton, usePanel } from "../panel-context";

const norm = (s: string) => s.trim().toLowerCase();

/** A number with its symbol and scale word set smaller beside it. */
export function BigValue({ parts }: { parts: ValueParts | null }) {
  if (!parts) return <span className="cv-nodata">—</span>;
  return (
    <>
      {parts.sign && <span className="cv-sign">{parts.sign}</span>}
      {parts.prefix && <span className="cv-pre">{parts.prefix}</span>}
      <span className="cv-num">{parts.number}</span>
      {parts.suffix && <span className="cv-suf" data-tight={parts.tight || undefined}>{parts.suffix}</span>}
    </>
  );
}

/** A change: arrow, signed value, what it is against; "calculated", its formula and its inputs underneath. */
export function Delta({ delta, unit, language }: { delta: DeltaSpec; unit: Unit | null; language: CanvasLanguage }) {
  const { labels } = usePanel();
  const dir = directionOf(delta.value);
  const text = formatDelta(delta.value, delta.kind, unit, language);
  const spoken = formatDelta(delta.value, delta.kind, unit, language, { spoken: true });
  const calc = delta.calculation;
  return (
    <div className="cv-delta" data-dir={dir}>
      <p className="cv-delta-main">
        <span className="cv-arrow" aria-hidden="true">
          {dir === "up" ? "▲" : dir === "down" ? "▼" : "▬"}
        </span>
        <span className="cv-delta-val" aria-label={`${spoken}, ${labels.calculated}${calc?.label ? `, ${calc.label}` : ""}`}>
          {text}
        </span>
        {calc?.label && <span className="cv-delta-label">{calc.label}</span>}
      </p>
      <p className="cv-delta-calc">
        <CalcBadge />
        {calc?.formula_text && <span className="cv-formula">{calc.formula_text}</span>}
      </p>
      {calc && calc.inputs.length > 0 && (
        <ul className="cv-inputs" aria-label={labels.inputs}>
          {calc.inputs.map((c, i) => (
            <li key={i}>
              <CellButton cell={c} className="cv-input">
                <span>{c.text || "·"}</span>
                {c.page !== null && <span className="cv-input-page">{`${labels.page} ${c.page}`}</span>}
              </CellButton>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function Kpi({ visual }: { visual: Visual }) {
  const lang = visual.language;
  const hl = new Set((visual.highlight?.x ?? []).map(norm));
  return (
    <ul className="cv-tiles" aria-label={visual.title}>
      {visual.tiles.map((t, i) => {
        const unit = t.unit ?? visual.unit;
        const strong = hl.has(norm(t.label));
        return (
          <li key={i} className="cv-tile" data-highlight={strong || undefined}>
            <p className="cv-tile-label">{t.label}</p>
            <p className="cv-tile-value">
              <CellButton cell={t.cell} ariaLabel={`${t.label}: ${formatValue(t.value, unit, lang, { spoken: true })}`}>
                <BigValue parts={formatParts(t.value, unit, lang)} />
              </CellButton>
            </p>
            {t.delta && <Delta delta={t.delta} unit={unit} language={lang} />}
            {strong && <span className="visually-hidden">{visual.highlight?.note ?? ""}</span>}
          </li>
        );
      })}
    </ul>
  );
}

/** The tile that carries the change for a metric: by name, or by position when there are as many tiles as metrics. */
function tileFor(tiles: Tile[], metric: string, index: number, count: number): Tile | null {
  return tiles.find((t) => norm(t.label) === norm(metric)) ?? (tiles.length === count ? tiles[index] : null) ?? null;
}

/**
 * A against B. The sides are the visual's series (FY23 | FY24), the metrics are its rows, and the change for each
 * metric is the tile with the same name. With no rows, the tiles themselves are listed with their changes.
 */
export function Comparison({ visual }: { visual: Visual }) {
  const { labels } = usePanel();
  const lang = visual.language;
  const sides = visual.series;
  const hl = new Set((visual.highlight?.x ?? []).map(norm));

  if (visual.rows.length === 0 || sides.length < 2) {
    return (
      <ul className="cv-compare cv-compare-list" aria-label={visual.title}>
        {visual.tiles.map((t, i) => {
          const unit = t.unit ?? visual.unit;
          return (
            <li key={i} className="cv-compare-row" data-highlight={hl.has(norm(t.label)) || undefined}>
              <p className="cv-compare-metric">{t.label}</p>
              <div className="cv-compare-cells">
                <p className="cv-compare-val">
                  <CellButton cell={t.cell}>
                    <BigValue parts={formatParts(t.value, unit, lang)} />
                  </CellButton>
                </p>
                {t.delta && <Delta delta={t.delta} unit={unit} language={lang} />}
              </div>
            </li>
          );
        })}
      </ul>
    );
  }

  return (
    <div className="cv-compare" role="group" aria-label={visual.title} style={{ "--sides": sides.length } as CSSProperties}>
      <div className="cv-compare-head" aria-hidden="true">
        <span />
        {sides.map((s) => (
          <span key={s.key} className="cv-side">
            {s.label}
          </span>
        ))}
        <span className="cv-side">Δ</span>
      </div>
      <ul className="cv-compare-body">
        {visual.rows.map((r, xi) => {
          const tile = tileFor(visual.tiles, r.x, xi, visual.rows.length);
          const strong = hl.has(norm(r.x));
          return (
            <li key={xi} className="cv-compare-row" data-highlight={strong || undefined}>
              <p className="cv-compare-metric">
                {r.x}
                {strong && <span className="visually-hidden">, {labels.highlighted}</span>}
              </p>
              <div className="cv-compare-cells">
                {sides.map((s) => {
                  const unit = s.unit ?? visual.unit;
                  const v = r.values[s.key] ?? null;
                  return (
                    <p key={s.key} className="cv-compare-val">
                      <span className="cv-side-inline" aria-hidden="true">
                        {s.label}
                      </span>
                      {v === null ? (
                        <span className="cv-nodata" aria-label={labels.noValue}>
                          —
                        </span>
                      ) : (
                        <CellButton cell={r.cells[s.key] ?? null} ariaLabel={`${r.x}, ${s.label}: ${formatValue(v, unit, lang, { spoken: true })}`}>
                          <BigValue parts={formatParts(v, unit, lang)} />
                        </CellButton>
                      )}
                    </p>
                  );
                })}
                {tile?.delta ? <Delta delta={tile.delta} unit={tile.unit ?? visual.unit} language={lang} /> : <span className="cv-delta-none" aria-hidden="true" />}
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
