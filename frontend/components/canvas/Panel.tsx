"use client";

/**
 * One visual on the canvas: title, subtitle, controls (View as table, pin, move up/down, remove, drag), the chart, the
 * highlight's note, how every calculated number was calculated, and the sources. The panel is an <article> named by its
 * title and described by the visual's `summary`, so a screen reader gets the text alternative before any data point.
 */

import { useId, useMemo, useRef, useState, type DragEvent } from "react";

import type { ProjectDocument } from "@/lib/api";
import { calcUnit, formatDate, formatValue, isIsoDate } from "@/lib/canvas/format";
import { buildModel, visualSourceRefs } from "@/lib/canvas/model";
import type { Visual } from "@/lib/canvas/types";

import { SourceList } from "../Citations";
import { Icon } from "../Icon";
import { CanvasChart, TableView } from "./charts";
import { EmptyNote } from "./chart-common";
import { CanvasIcon } from "./icons";
import { CalcBadge, CellButton, PanelProvider, usePanel } from "./panel-context";

export type PanelAction = "pin" | "unpin" | "remove" | "up" | "down";

export interface PanelProps {
  visual: Visual;
  docsById: Record<string, ProjectDocument>;
  index: number;
  count: number;
  /** An operation on this panel is in flight. */
  busy?: boolean;
  /** The backend is rebuilding this visual (a `preparing` event for an id that is already on the canvas). */
  updating?: boolean;
  /** The project overview: no pin, move or remove. */
  readOnly?: boolean;
  onAction?: (action: PanelAction, visualId: string) => void;
  dragging?: boolean;
  dropTarget?: boolean;
  onDragStart?: (visualId: string) => void;
  onDragEnd?: () => void;
  onDragEnter?: (visualId: string) => void;
  onDrop?: (visualId: string) => void;
}

export function Panel(props: PanelProps) {
  return (
    <PanelProvider visual={props.visual} docsById={props.docsById}>
      <PanelInner {...props} />
    </PanelProvider>
  );
}

function PanelInner({ visual, docsById, index, count, busy, updating, readOnly, onAction, dragging, dropTarget, onDragStart, onDragEnd, onDragEnter, onDrop }: PanelProps) {
  const { labels } = usePanel();
  const titleId = useId();
  const summaryId = useId();
  const [asTable, setAsTable] = useState(false);
  const articleRef = useRef<HTMLElement>(null);
  const model = useMemo(() => buildModel(visual), [visual]);
  const refs = useMemo(() => visualSourceRefs(visual, docsById), [visual, docsById]);
  const kind = visual.kind;
  const canToggle = kind !== "table";
  const highlight = visual.highlight;

  const dragProps =
    readOnly || !onDragStart
      ? {}
      : {
          onDragOver: (e: DragEvent) => {
            if (dragging === undefined || !Array.from(e.dataTransfer.types).includes("application/x-canvas-panel")) return;
            e.preventDefault();
            e.dataTransfer.dropEffect = "move";
          },
          onDragEnter: (e: DragEvent) => {
            if (Array.from(e.dataTransfer.types).includes("application/x-canvas-panel")) onDragEnter?.(visual.id);
          },
          onDrop: (e: DragEvent) => {
            e.preventDefault();
            onDrop?.(visual.id);
          },
        };

  return (
    <article
      ref={articleRef}
      className="cv-panel"
      data-kind={kind}
      data-pinned={visual.pinned || undefined}
      data-busy={busy || updating || undefined}
      data-dragging={dragging || undefined}
      data-drop={dropTarget || undefined}
      data-panel={visual.id}
      lang={visual.language}
      aria-labelledby={titleId}
      aria-describedby={summaryId}
      aria-busy={busy || updating || undefined}
      {...dragProps}
    >
      <header className="cv-head">
        <div className="cv-head-text">
          <h3 id={titleId} className="cv-title">
            {visual.title}
            {visual.pinned && (
              <span className="cv-pinned" title={labels.pinned}>
                <CanvasIcon name="pin" size={13} filled />
                <span className="visually-hidden">, {labels.pinned}</span>
              </span>
            )}
          </h3>
          {visual.subtitle && <p className="cv-subtitle">{visual.subtitle}</p>}
        </div>
        <div className="cv-actions" role="group" aria-label={`${visual.title}: actions`}>
          {canToggle && (
            <button type="button" className="btn btn-sm cv-view" aria-pressed={asTable} onClick={() => setAsTable((v) => !v)}>
              <CanvasIcon name={asTable ? "chart" : "table"} size={14} />
              {asTable ? labels.viewChart : labels.viewTable}
            </button>
          )}
          {!readOnly && onAction && (
            <>
              <button
                type="button"
                className="icon-btn icon-btn-sm"
                aria-pressed={visual.pinned}
                aria-label={visual.pinned ? labels.unpin : labels.pin}
                title={visual.pinned ? labels.unpin : labels.pin}
                aria-disabled={busy || undefined}
                onClick={() => !busy && onAction(visual.pinned ? "unpin" : "pin", visual.id)}
              >
                <CanvasIcon name="pin" size={15} filled={visual.pinned} />
              </button>
              <button
                type="button"
                className="icon-btn icon-btn-sm"
                data-move="up"
                aria-label={`${labels.moveUp} (${labels.position(index + 1, count)})`}
                title={labels.moveUp}
                aria-disabled={busy || index === 0 || undefined}
                onClick={() => !busy && index > 0 && onAction("up", visual.id)}
              >
                <Icon name="arrowUp" size={15} />
              </button>
              <button
                type="button"
                className="icon-btn icon-btn-sm"
                data-move="down"
                aria-label={`${labels.moveDown} (${labels.position(index + 1, count)})`}
                title={labels.moveDown}
                aria-disabled={busy || index === count - 1 || undefined}
                onClick={() => !busy && index < count - 1 && onAction("down", visual.id)}
              >
                <Icon name="arrowDown" size={15} />
              </button>
              <button
                type="button"
                className="icon-btn icon-btn-sm cv-grip"
                draggable={count > 1}
                aria-hidden={count > 1 ? undefined : true}
                tabIndex={-1}
                title={labels.drag}
                onDragStart={(e) => {
                  e.dataTransfer.setData("application/x-canvas-panel", visual.id);
                  e.dataTransfer.effectAllowed = "move";
                  if (articleRef.current) e.dataTransfer.setDragImage(articleRef.current, 24, 18);
                  onDragStart?.(visual.id);
                }}
                onDragEnd={() => onDragEnd?.()}
              >
                <CanvasIcon name="grip" size={15} />
                <span className="visually-hidden">{labels.drag}</span>
              </button>
              <button
                type="button"
                className="icon-btn icon-btn-sm cv-remove"
                aria-label={labels.remove}
                title={labels.remove}
                aria-disabled={busy || undefined}
                onClick={() => !busy && onAction("remove", visual.id)}
              >
                <Icon name="close" size={15} />
              </button>
            </>
          )}
        </div>
      </header>

      <p id={summaryId} className={asTable ? "cv-summary" : "visually-hidden"}>
        {visual.summary}
      </p>
      {visual.requested_kind && <p className="cv-note">{labels.unsupported(visual.requested_kind)}</p>}
      {updating && (
        <p className="cv-updating" role="status">
          {labels.updating}
        </p>
      )}

      <div className="cv-body">{asTable ? <TableView visual={visual} model={model} /> : <PanelChart visual={visual} model={model} summaryId={summaryId} />}</div>

      {highlight?.note && (
        <p className="cv-highlight">
          <span className="cv-flag-key" aria-hidden="true">
            ▾
          </span>
          <span>
            <strong>{labels.highlighted}</strong>
            {highlight.x.length > 0
              ? ` (${highlight.x.map((x) => (isIsoDate(x) ? formatDate(x, visual.language) : x)).join(", ")})`
              : highlight.series.length > 0
                ? ` (${highlight.series.join(", ")})`
                : ""}
            {": "}
            {highlight.note}
          </span>
        </p>
      )}

      <Calculations visual={visual} />

      {refs.length > 0 && (
        <footer className="cv-sources">
          <span className="cv-sources-label">{labels.sources}</span>
          <SourceList sources={refs} />
        </footer>
      )}
    </article>
  );
}

function PanelChart({ visual, model, summaryId }: { visual: Visual; model: ReturnType<typeof buildModel>; summaryId: string }) {
  const empty =
    visual.kind === "kpi" || visual.kind === "comparison"
      ? visual.tiles.length === 0 && visual.rows.length === 0
      : visual.kind === "timeline"
        ? visual.events.length === 0
        : visual.rows.length === 0;
  if (empty) return <EmptyNote />;
  return <CanvasChart visual={visual} model={model} summaryId={summaryId} />;
}

/** Every derived number in the visual, how it was worked out and the cells it started from. */
function Calculations({ visual }: { visual: Visual }) {
  const { labels } = usePanel();
  const lang = visual.language;
  if (visual.calculations.length === 0) return null;
  return (
    <details className="cv-calcs">
      <summary>
        <CanvasIcon name="formula" size={14} />
        {labels.calculations} ({visual.calculations.length})
      </summary>
      <ul className="cv-calc-list">
        {visual.calculations.map((c, i) => (
          <li key={i}>
            <p className="cv-calc-head">
              <span className="cv-calc-label">{c.label}</span>
              <strong className="cv-calc-value">{formatValue(c.value, calcUnit(c.op, c.unit, visual.unit), lang)}</strong>
              <CalcBadge />
            </p>
            {c.formula_text && (
              <p className="cv-calc-formula">
                <span>{labels.formula}</span> <code>{c.formula_text}</code>
              </p>
            )}
            {c.inputs.length > 0 && (
              <ul className="cv-inputs" aria-label={labels.inputs}>
                {c.inputs.map((cell, j) => (
                  <li key={j}>
                    <CellButton cell={cell} className="cv-input">
                      <span>{cell.text || "·"}</span>
                      {cell.page !== null && <span className="cv-input-page">{`${labels.page} ${cell.page}`}</span>}
                    </CellButton>
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </details>
  );
}
