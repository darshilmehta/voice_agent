"use client";

/**
 * The panels of a canvas or overview, with every chart. This module is the lazy chunk: the page loads it only when a
 * canvas or an overview has something to draw (see CanvasPanel.tsx), so chats without visuals never download chart code.
 *
 * It renders panels as grid items (a fragment), inside the grid its caller owns, and provides the citation popover
 * that data points, tiles and table cells open for their source.
 */

import { useCallback, useEffect, useRef, useState } from "react";

// The panel and chart styles travel with the chart code: a page without visuals never downloads them.
import "@/app/styles/canvas-panels.css";

import type { ProjectDocument } from "@/lib/api";
import { shellLabelsFor } from "@/lib/canvas/labels";
import type { CanvasOp, Visual } from "@/lib/canvas/types";

import { CitationPopoverProvider } from "../Citations";
import { CanvasBoundary } from "./Boundary";
import { Panel, type PanelAction } from "./Panel";

export interface BoardProps {
  panels: Visual[];
  /** Ids of panels being rebuilt (a `preparing` event for a visual that is already here). */
  updating: ReadonlySet<string>;
  /** Ids with an operation in flight. */
  busy: ReadonlySet<string>;
  docsById: Record<string, ProjectDocument>;
  readOnly?: boolean;
  onOp?: (op: CanvasOp, visualId: string, position?: number) => Promise<boolean>;
}

export default function CanvasBoard({ panels, updating, busy, docsById, readOnly, onOp }: BoardProps) {
  const [dragId, setDragId] = useState<string | null>(null);
  const [overId, setOverId] = useState<string | null>(null);
  // After a keyboard move, focus stays on the button that was used, wherever the panel ends up. After a removal it goes
  // to the panel that took its place, or back to the microphone when the canvas is empty: never to the top of the page.
  const focusAfter = useRef<{ id: string; dir: "up" | "down" } | { id: string | null; removed: true } | null>(null);

  const order = panels.map((p) => p.id).join("|");
  useEffect(() => {
    const f = focusAfter.current;
    if (!f) return;
    if ("removed" in f) {
      const next = f.id ? document.querySelector(`[data-panel="${CSS.escape(f.id)}"]`) : null;
      focusAfter.current = null;
      (next?.querySelector<HTMLElement>(".cv-actions button") ?? document.querySelector<HTMLElement>(".vc-mic"))?.focus();
      return;
    }
    focusAfter.current = null;
    const root = document.querySelector(`[data-panel="${CSS.escape(f.id)}"]`);
    const wanted = root?.querySelector<HTMLElement>(`[data-move="${f.dir}"]`);
    const other = root?.querySelector<HTMLElement>(`[data-move="${f.dir === "up" ? "down" : "up"}"]`);
    (wanted?.getAttribute("aria-disabled") ? other : wanted)?.focus();
  }, [order]);

  const onAction = useCallback(
    (action: PanelAction, id: string) => {
      if (!onOp) return;
      const at = panels.findIndex((p) => p.id === id);
      if (at < 0) return;
      if (action === "up" || action === "down") {
        const target = panels[action === "up" ? at - 1 : at + 1];
        if (!target) return;
        focusAfter.current = { id, dir: action };
        void onOp("move", id, target.position).then((ok) => {
          if (!ok) focusAfter.current = null;
        });
        return;
      }
      if (action === "remove") {
        const nextId = (panels[at + 1] ?? panels[at - 1])?.id ?? null;
        focusAfter.current = { id: nextId, removed: true };
        void onOp("remove", id).then((ok) => {
          if (!ok) focusAfter.current = null;
          // The last visual is gone and the canvas with it: focus goes back to the microphone.
          else if (!nextId) window.setTimeout(() => document.querySelector<HTMLElement>(".vc-mic")?.focus(), 0);
        });
        return;
      }
      void onOp(action, id);
    },
    [onOp, panels],
  );

  const onDrop = useCallback(
    (targetId: string) => {
      const from = dragId;
      setDragId(null);
      setOverId(null);
      if (!from || from === targetId || !onOp) return;
      const target = panels.find((p) => p.id === targetId);
      if (target) void onOp("move", from, target.position);
    },
    [dragId, onOp, panels],
  );

  return (
    <CitationPopoverProvider>
      {panels.map((visual, index) => (
        <CanvasBoundary
          key={visual.id}
          fallback={(retry) => (
            <article className="cv-panel" data-panel={visual.id} lang={visual.language}>
              <header className="cv-head">
                <h3 className="cv-title">{visual.title}</h3>
              </header>
              <p className="cv-note">
                {shellLabelsFor(visual.language).failed}{" "}
                <button type="button" className="link-btn" onClick={retry}>
                  {shellLabelsFor(visual.language).retry}
                </button>
              </p>
            </article>
          )}
        >
          <Panel
            visual={visual}
            docsById={docsById}
            index={index}
            count={panels.length}
            busy={busy.has(visual.id)}
            updating={updating.has(visual.id)}
            readOnly={readOnly}
            onAction={onAction}
            dragging={dragId === visual.id ? true : dragId ? false : undefined}
            dropTarget={dragId !== null && overId === visual.id && dragId !== visual.id}
            onDragStart={setDragId}
            onDragEnd={() => {
              setDragId(null);
              setOverId(null);
            }}
            onDragEnter={setOverId}
            onDrop={onDrop}
          />
        </CanvasBoundary>
      ))}
    </CitationPopoverProvider>
  );
}
