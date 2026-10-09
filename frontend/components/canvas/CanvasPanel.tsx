"use client";

/**
 * The canvas shell on a chat page (docs/DESIGN.md §12.1). Light by design: it knows the canvas state, draws the
 * skeletons and quiet failure notes, and loads the chart code (CanvasBoard) lazily, only once there is a panel to draw
 * or one on its way. With nothing on the canvas the page doesn't render this at all (no chrome, no empty state).
 *
 *   preparing → a skeleton where the visual will appear (and the chart chunk starts loading)
 *   ready     → the panel, in canvas order
 *   failed    → a quiet note that goes away by itself; the answer is unaffected
 *
 * At narrow widths the panels become a swipeable row (scroll-snap) with dots, a sheet above the voice controls.
 */

import { lazy, Suspense, useEffect, useRef, useState } from "react";

import type { ProjectDocument } from "@/lib/api";
import { shellLabelsFor } from "@/lib/canvas/labels";
import { SKELETON_DELAY_MS } from "@/lib/canvas/state";
import type { CanvasController } from "@/lib/canvas/use-canvas";

import { Icon } from "../Icon";
import { CanvasBoundary } from "./Boundary";

/** More panels than this and the swipe indicator is a count ("3 of 15") instead of dots. */
const MAX_DOTS = 8;

const loadBoard = () => import("./CanvasBoard");
const LazyBoard = lazy(loadBoard);

export function CanvasPanel({
  canvas,
  docsById,
  language,
}: {
  canvas: CanvasController;
  docsById: Record<string, ProjectDocument>;
  language: string | null;
}) {
  const labels = shellLabelsFor(language);
  const { panels, pending, busy, op, dismiss } = canvas;
  const onCanvas = new Set(panels.map((p) => p.id));
  const updating = new Set(pending.filter((p) => p.phase === "preparing" && onCanvas.has(p.id)).map((p) => p.id));
  const fresh = pending.filter((p) => !onCanvas.has(p.id));
  const preparing = fresh.filter((p) => p.phase === "preparing");

  // The chart code starts loading as soon as a visual is on its way, so the skeleton becomes the chart without a second wait.
  const anyPreparing = canvas.preparing || pending.some((p) => p.phase === "preparing");
  useEffect(() => {
    if (anyPreparing) void loadBoard();
  }, [anyPreparing]);

  // A skeleton on screen stays until its chart can replace it (the chart code may still be loading when `ready` comes).
  const skeletons = useRef(new Set<string>());
  useEffect(() => {
    for (const p of preparing) skeletons.current.add(p.id);
  });

  const gridRef = useRef<HTMLDivElement>(null);
  const [page, setPage] = useState(0);
  const total = panels.length + preparing.length;
  useEffect(() => {
    const el = gridRef.current;
    if (!el) return;
    const onScroll = () => {
      if (el.scrollWidth <= el.clientWidth + 4) return;
      setPage(Math.min(total - 1, Math.max(0, Math.round(el.scrollLeft / Math.max(1, el.clientWidth)))));
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [total]);

  const goTo = (i: number) => {
    const el = gridRef.current;
    const child = el?.children[i] as HTMLElement | undefined;
    if (!el || !child) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    el.scrollTo({ left: child.offsetLeft - el.offsetLeft, behavior: reduce ? "auto" : "smooth" });
  };

  return (
    <section className="cv-canvas" aria-label={labels.visuals} data-count={total}>
      <div ref={gridRef} className="cv-board">
        {panels.length > 0 && (
          <CanvasBoundary
            fallback={() => (
              <p className="cv-note cv-chunk-failed">
                {labels.failed}{" "}
                <button type="button" className="link-btn" onClick={() => window.location.reload()}>
                  <Icon name="refresh" size={13} /> Reload
                </button>
              </p>
            )}
          >
            <Suspense
              fallback={panels.map((p) => (
                <DelayedSkeleton key={p.id} text={labels.preparing} immediate={skeletons.current.has(p.id)} />
              ))}
            >
              <LazyBoard panels={panels} updating={updating} busy={busy} docsById={docsById} onOp={op} />
            </Suspense>
          </CanvasBoundary>
        )}
        {fresh.map((p) =>
          p.phase === "preparing" ? (
            <PanelSkeleton key={p.id} text={labels.preparing} />
          ) : (
            <div key={p.id} className="cv-failed" role="status">
              <Icon name="info" size={14} />
              <span>
                {labels.failed}
                {p.detail ? ` (${p.detail})` : ""}
              </span>
              <button type="button" className="icon-btn icon-btn-sm" aria-label={labels.dismiss} onClick={() => dismiss(p.id)}>
                <Icon name="close" size={13} />
              </button>
            </div>
          ),
        )}
      </div>
      {total > 1 && total <= MAX_DOTS && (
        <div className="cv-dots" role="group" aria-label={labels.visuals}>
          {Array.from({ length: total }, (_, i) => (
            <button
              key={i}
              type="button"
              className="cv-dot-btn"
              aria-label={`${labels.panel} ${labels.position(i + 1, total)}`}
              aria-current={i === page || undefined}
              onClick={() => goTo(i)}
            />
          ))}
        </div>
      )}
      {total > MAX_DOTS && (
        <p className="cv-dots cv-dots-count" aria-hidden="true">
          {labels.position(page + 1, total)}
        </p>
      )}
    </section>
  );
}

/**
 * A skeleton that appears only if what it stands for hasn't arrived after SKELETON_DELAY_MS: the fallback of the lazy
 * chart code, which is usually there within a frame (a chunk already fetched, or fetched while the draft was being
 * made) and would otherwise flash a placeholder in the panel's place. `immediate`: a skeleton of this visual is on
 * screen already, and stays until the chart replaces it (no blank frame between the two).
 */
export function DelayedSkeleton({ text, immediate = false }: { text: string; immediate?: boolean }) {
  const [shown, setShown] = useState(immediate);
  useEffect(() => {
    if (immediate) return;
    const timer = window.setTimeout(() => setShown(true), SKELETON_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [immediate]);
  return shown ? <PanelSkeleton text={text} /> : null;
}

/** What a visual looks like while it is being prepared: a card with a title line and quiet bars, never a spinner. */
export function PanelSkeleton({ text }: { text: string }) {
  return (
    <article className="cv-panel cv-skeleton" aria-busy="true">
      <div className="skel cv-skel-title" />
      <div className="cv-skel-chart" aria-hidden="true">
        {[46, 62, 40, 74, 58, 82].map((h, i) => (
          <span key={i} className="skel" style={{ height: `${h}%` }} />
        ))}
      </div>
      <p className="cv-skel-text" role="status">
        {text}
      </p>
    </article>
  );
}
