"use client";

/**
 * What every part of one panel shares: the visual, its words (in its own language), the project's documents, and how
 * to resolve a cell to its source and open the existing citation popover for it (components/Citations.tsx).
 */

import { createContext, useCallback, useContext, useId, useMemo, useRef, type ReactNode } from "react";

import type { ProjectDocument } from "@/lib/api";
import { resolveCell, type CellSource } from "@/lib/canvas/model";
import { labelsFor, type Labels } from "@/lib/canvas/labels-panel";
import type { CellRef, Visual } from "@/lib/canvas/types";

import { usePopover } from "../Citations";
import { CanvasIcon } from "./icons";

export interface PanelApi {
  visual: Visual;
  labels: Labels;
  docsById: Record<string, ProjectDocument>;
  resolve: (cell: CellRef) => CellSource;
  /** Toggle the citation popover for `cell`, anchored on `el`. */
  open: (el: HTMLElement, cell: CellRef) => void;
}

const PanelContext = createContext<PanelApi | null>(null);

export function PanelProvider({
  visual,
  docsById,
  children,
}: {
  visual: Visual;
  docsById: Record<string, ProjectDocument>;
  children: ReactNode;
}) {
  const pop = usePopover();
  const labels = labelsFor(visual.language);
  const resolve = useCallback((cell: CellRef) => resolveCell(visual, cell, docsById), [visual, docsById]);
  const open = useCallback((el: HTMLElement, cell: CellRef) => pop.toggle(el, resolve(cell).ref), [pop, resolve]);
  const value = useMemo<PanelApi>(() => ({ visual, labels, docsById, resolve, open }), [visual, labels, docsById, resolve, open]);
  return <PanelContext value={value}>{children}</PanelContext>;
}

export function usePanel(): PanelApi {
  const ctx = useContext(PanelContext);
  if (!ctx) throw new Error("canvas parts must be inside <PanelProvider>");
  return ctx;
}

/** "calculated" with its formula glyph: marks every derived number. */
export function CalcBadge({ className }: { className?: string }) {
  const { labels } = usePanel();
  return (
    <span className={className ? `cv-calc ${className}` : "cv-calc"} title={labels.calculatedHint}>
      <CanvasIcon name="formula" size={11} />
      {labels.calculated}
    </span>
  );
}

/**
 * A figure that came from a cell: hovering, focusing or clicking it opens the citation popover for its source (the
 * same popover as the `[S1]` chips), which shows the document, the page and the cell's text. Without a cell it is
 * plain text.
 */
export function CellButton({
  cell,
  children,
  className,
  ariaLabel,
}: {
  cell: CellRef | null;
  children: ReactNode;
  className?: string;
  ariaLabel?: string;
}) {
  const { resolve } = usePanel();
  const pop = usePopover();
  const ref = useRef<HTMLButtonElement>(null);
  const id = useId();
  if (!cell) return <span className={className}>{children}</span>;
  const source = resolve(cell);
  const el = () => ref.current;
  return (
    <button
      ref={ref}
      id={id}
      type="button"
      className={className ? `cv-cell ${className}` : "cv-cell"}
      aria-label={ariaLabel}
      aria-describedby={pop.describedBy(id)}
      onClick={() => el() && pop.toggle(el()!, source.ref)}
      onFocus={() => {
        const b = el();
        if (b && b.matches(":focus-visible")) pop.openFor(b, source.ref, "focus");
      }}
      onBlur={() => el() && pop.closeFor(el()!, ["focus", "click"])}
      onPointerMove={(e) => {
        const b = el();
        if (e.pointerType === "mouse" && b && pop.current !== b) pop.openFor(b, source.ref, "hover");
      }}
      onMouseLeave={() => el() && pop.hoverOut(el()!)}
    >
      {children}
    </button>
  );
}
