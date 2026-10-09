"use client";

/**
 * The interactive layer every drawn chart shares (docs/DESIGN.md §12.1; the dataviz interaction rules):
 *
 *   - the marks are SVG and decorative to assistive technology; over them sits one real <button> per data point (at
 *     least 24 px, bigger than the mark), so every point can be focused, announced and clicked;
 *   - the buttons are one tab stop (roving): arrow keys move between positions and series, Home/End jump, and Enter
 *     opens the source in the citation popover;
 *   - hovering or focusing a point shows a tooltip with its value, label and source (document, page, the cell's
 *     text), calculated values are marked and show their formula; on a line chart the pointer snaps to the nearest
 *     position, and the tooltip lists every series there;
 *   - a tooltip never gates information: every value is also in the table view and in the buttons' accessible names.
 */

import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type FocusEvent as ReactFocusEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from "react";

import { moveCursor, placeTooltip, type Cursor } from "@/lib/canvas/scales";
import type { CellRef } from "@/lib/canvas/types";

import { usePopover } from "../Citations";
import { CanvasIcon } from "./icons";
import { CalcBadge, usePanel } from "./panel-context";

export interface HitBox {
  xi: number;
  si: number;
  /** The clickable area, in chart pixels (bigger than the mark). */
  x: number;
  y: number;
  w: number;
  h: number;
  /** The mark this point is drawn as: where the focus ring goes. */
  mark: { x: number; y: number; w: number; h: number; shape: "rect" | "circle" };
  /** Where the tooltip points. */
  ax: number;
  ay: number;
}

export interface TipRow {
  color: string;
  shape: "line" | "bar" | "dot";
  label: string;
  value: string;
  calculated: boolean;
  active: boolean;
}

export interface Tip {
  title: string;
  rows: TipRow[];
  /** The active row's provenance. */
  source: { filename: string; page: number | null; cellText: string } | null;
  /** The active row's formula, when its number is calculated. */
  formula: string | null;
}

export interface Described {
  /** The accessible name of the point's button. */
  aria: string;
  tip: Tip;
  cell: CellRef | null;
}

type Via = "pointer" | "focus" | "touch";
interface Active extends Cursor {
  via: Via;
}

export interface FrameChildren {
  active: Cursor | null;
  /** Pointer handlers for marks drawn in the SVG (a donut slice): hovering them targets that point, clicking opens it. */
  bind: (xi: number, si: number) => {
    onPointerEnter: (e: ReactPointerEvent) => void;
    onPointerLeave: (e: ReactPointerEvent) => void;
    onClick: () => void;
  };
}

interface Props {
  width: number;
  height: number;
  hits: HitBox[];
  /** Grid size for arrow keys: positions × series. */
  nx: number;
  ns: number;
  /** Categories run down the page (sideways bars): up/down change position. */
  sideways?: boolean;
  label: string;
  describedBy?: string;
  describe: (xi: number, si: number) => Described;
  /** A line chart: the pointer snaps to the nearest of these x positions (plot area from x0 to x1). */
  snap?: { xs: number[]; x0: number; x1: number };
  /** Drawn over the marks inside the SVG while a point is active (the crosshair). */
  overlay?: (active: Cursor | null) => ReactNode;
  children: (c: FrameChildren) => ReactNode;
}

const key = (xi: number, si: number) => `${xi}:${si}`;

export function ChartFrame({ width, height, hits, nx, ns, sideways, label, describedBy, describe, snap, overlay, children }: Props) {
  const { open, labels } = usePanel();
  const pop = usePopover();
  const uid = useId();
  const hintId = `${uid}hint`;
  const plotRef = useRef<HTMLDivElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const buttons = useRef(new Map<string, HTMLButtonElement>());
  const [active, setActive] = useState<Active | null>(null);
  const [rove, setRove] = useState<Cursor | null>(null);
  const [tipSize, setTipSize] = useState<{ w: number; h: number } | null>(null);

  const hitMap = useMemo(() => new Map(hits.map((h) => [key(h.xi, h.si), h])), [hits]);
  const first = hits[0];
  const roving = rove && hitMap.has(key(rove.xi, rove.si)) ? rove : first ? { xi: first.xi, si: first.si } : null;

  // Pointer moved out, or the data changed under it: a cursor on a point that no longer exists goes away.
  const live = active && hitMap.has(key(active.xi, active.si)) ? active : null;

  const described = useMemo(() => (live ? describe(live.xi, live.si) : null), [live, describe]);
  const hit = live ? hitMap.get(key(live.xi, live.si)) : undefined;
  const activeButton = live ? buttons.current.get(key(live.xi, live.si)) : undefined;
  // The citation popover is open on this point: it says the same thing and more, so the tooltip steps aside.
  const popoverOnPoint = !!activeButton && pop.current === activeButton;
  const showTip = !!live && !!described && !!hit && !popoverOnPoint;

  useLayoutEffect(() => {
    const el = tipRef.current;
    if (!showTip || !el) return;
    const r = el.getBoundingClientRect();
    setTipSize((s) => (s && Math.abs(s.w - r.width) < 0.5 && Math.abs(s.h - r.height) < 0.5 ? s : { w: r.width, h: r.height }));
  });

  // A tap leaves the tooltip up; the next tap elsewhere takes it down.
  const touching = active?.via === "touch";
  useEffect(() => {
    if (!touching) return;
    const onDown = (e: PointerEvent) => {
      if (!plotRef.current?.contains(e.target as Node)) setActive(null);
    };
    document.addEventListener("pointerdown", onDown, true);
    return () => document.removeEventListener("pointerdown", onDown, true);
  }, [touching]);

  const activate = useCallback((xi: number, si: number, via: Via) => {
    setActive((cur) => (cur && cur.xi === xi && cur.si === si && cur.via === via ? cur : { xi, si, via }));
  }, []);

  const clear = useCallback((via: Via) => setActive((cur) => (cur && cur.via === via ? null : cur)), []);

  const onPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!snap || e.pointerType === "touch") return;
    const rect = plotRef.current?.getBoundingClientRect();
    if (!rect) return;
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    if (px < snap.x0 - 16 || px > snap.x1 + 16) return clear("pointer");
    let xi = 0;
    let best = Infinity;
    snap.xs.forEach((x, i) => {
      const d = Math.abs(x - px);
      if (d < best) {
        best = d;
        xi = i;
      }
    });
    // Within that position, the series whose point is nearest the pointer's height.
    let si = -1;
    let bestY = Infinity;
    for (let s = 0; s < ns; s++) {
      const h = hitMap.get(key(xi, s));
      if (!h) continue;
      const d = Math.abs(h.mark.y + h.mark.h / 2 - py);
      if (d < bestY) {
        bestY = d;
        si = s;
      }
    }
    if (si >= 0) activate(xi, si, "pointer");
  };

  const onKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Escape") {
      setActive(null);
      return;
    }
    const target = e.target as HTMLElement;
    if (target.dataset.xi === undefined) return;
    const cur = { xi: Number(target.dataset.xi), si: Number(target.dataset.si) };
    let next = moveCursor(cur, e.key, nx, ns, { vertical: sideways });
    // Skip positions with no point (a series that has none there) in the same direction.
    for (let guard = 0; next && !hitMap.has(key(next.xi, next.si)) && guard < nx + ns; guard++) {
      next = moveCursor(next, e.key, nx, ns, { vertical: sideways });
    }
    if (!next) return;
    e.preventDefault();
    buttons.current.get(key(next.xi, next.si))?.focus();
  };

  const onBlur = (e: ReactFocusEvent<HTMLDivElement>) => {
    if (!e.currentTarget.contains(e.relatedTarget as Node | null)) clear("focus");
  };

  const placement = showTip && tipSize && hit ? placeTooltip({ x: hit.ax, y: hit.ay }, tipSize, { w: width, h: height }) : null;

  const ring = live?.via === "focus" && hit ? hit.mark : null;

  const bind: FrameChildren["bind"] = (xi, si) => ({
    onPointerEnter: (e) => activate(xi, si, e.pointerType === "touch" ? "touch" : "pointer"),
    onPointerLeave: (e) => e.pointerType !== "touch" && clear("pointer"),
    onClick: () => buttons.current.get(key(xi, si))?.click(),
  });

  return (
    <div
      ref={plotRef}
      className="cv-plot"
      style={{ width, height }}
      onPointerMove={onPointerMove}
      onPointerLeave={(e) => e.pointerType !== "touch" && clear("pointer")}
    >
      <svg className="cv-svg" width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true" focusable="false">
        {children({ active: live, bind })}
        {overlay?.(live)}
        {ring &&
          (ring.shape === "circle" ? (
            <circle className="cv-focus" cx={ring.x + ring.w / 2} cy={ring.y + ring.h / 2} r={ring.w / 2 + 3} />
          ) : (
            <rect className="cv-focus" x={ring.x - 3} y={ring.y - 3} width={ring.w + 6} height={ring.h + 6} rx={4} />
          ))}
      </svg>

      <div
        className="cv-hits"
        role="group"
        aria-label={label}
        aria-describedby={describedBy ? `${describedBy} ${hintId}` : hintId}
        onKeyDown={onKeyDown}
        onBlur={onBlur}
      >
        <span id={hintId} className="visually-hidden">
          {labels.chartKeys}
        </span>
        {hits.map((h) => {
          const d = describe(h.xi, h.si);
          const isRoving = roving?.xi === h.xi && roving?.si === h.si;
          return (
            <button
              key={key(h.xi, h.si)}
              ref={(el) => {
                if (el) buttons.current.set(key(h.xi, h.si), el);
                else buttons.current.delete(key(h.xi, h.si));
              }}
              type="button"
              className="cv-hit"
              data-xi={h.xi}
              data-si={h.si}
              data-round={h.mark.shape === "circle" || undefined}
              tabIndex={isRoving ? 0 : -1}
              aria-label={d.aria}
              aria-describedby={pop.describedBy(`${uid}${key(h.xi, h.si)}`)}
              id={`${uid}${key(h.xi, h.si)}`}
              style={{ left: h.x, top: h.y, width: h.w, height: h.h }}
              onFocus={(e) => {
                setRove({ xi: h.xi, si: h.si });
                if (e.currentTarget.matches(":focus-visible")) activate(h.xi, h.si, "focus");
              }}
              onPointerEnter={(e) => activate(h.xi, h.si, e.pointerType === "touch" ? "touch" : "pointer")}
              onPointerLeave={(e) => e.pointerType !== "touch" && !snap && clear("pointer")}
              onClick={(e) => {
                if (d.cell) open(e.currentTarget, d.cell);
              }}
            />
          );
        })}
      </div>

      {showTip && described && (
        <div
          ref={tipRef}
          className="cv-tip"
          role="presentation"
          aria-hidden="true"
          data-side={placement?.side}
          style={placement ? { left: placement.left, top: placement.top } : { left: 0, top: 0, visibility: "hidden" }}
        >
          <TipCard tip={described.tip} hasSource={!!described.cell} />
        </div>
      )}
    </div>
  );
}

function TipCard({ tip, hasSource }: { tip: Tip; hasSource: boolean }) {
  const { labels } = usePanel();
  return (
    <>
      <p className="cv-tip-x">{tip.title}</p>
      <ul className="cv-tip-rows">
        {tip.rows.map((r, i) => (
          <li key={i} data-active={r.active || undefined}>
            <span className="cv-key" data-shape={r.shape} style={{ background: r.color }} />
            <span className="cv-tip-val">{r.value}</span>
            <span className="cv-tip-name">{r.label}</span>
            {r.calculated && <CalcBadge />}
          </li>
        ))}
      </ul>
      {tip.formula && (
        <p className="cv-tip-formula">
          <CanvasIcon name="formula" size={12} />
          <span>{tip.formula}</span>
        </p>
      )}
      {tip.source && (
        <p className="cv-tip-src">
          <span className="cv-tip-file">{tip.source.filename}</span>
          {tip.source.page !== null && (
            <span>
              {" · "}
              {labels.page} {tip.source.page}
            </span>
          )}
          {tip.source.cellText && <span>{` · ${labels.cell} “${tip.source.cellText}”`}</span>}
        </p>
      )}
      {hasSource && <p className="cv-tip-hint">{labels.clickForSource}</p>}
    </>
  );
}
