"use client";

/**
 * Citations in answers: `[S1]` markers (documents) and `[W1]` markers (live web results, docs/DESIGN.md §3.7) become
 * small chips inside the text, each message lists the sources it cites (documents first, then web), and hovering,
 * focusing or clicking a chip opens one shared popover with the document, page(s) and the cited passage, or the web
 * result's title, site, date and a link to the page.
 *
 * The popover is a tooltip-style description of the chip (role="tooltip", linked with aria-describedby): keyboard
 * focus or hover shows it, a click pins it open until you click again, press Esc, move focus away or click
 * elsewhere. It follows the chip when the transcript scrolls. A web result's popover holds a link, so it is a small
 * non-modal dialog instead: with the keyboard, Tab from the chip moves into the link (the popover sits at the end of
 * the page, outside the chip's tab order), Tab or Shift+Tab from the link goes back to the chip, and Esc closes it
 * and returns to the chip. The link opens the page in a new tab (`noopener noreferrer`) and only ever for http(s).
 */

import {
  createContext,
  useCallback,
  useContext,
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
import { createPortal } from "react-dom";

import {
  describeSource,
  hostOf,
  isNumericCell,
  locationShort,
  markerOf,
  pagesLong,
  parseTableSnippet,
  publishedText,
  sectionPath,
  sectionShort,
  sortSources,
  splitCitations,
  withoutPartialMarker,
  type SourceRef,
} from "@/lib/citations";

import { Icon } from "./Icon";

type Mode = "hover" | "focus" | "click";

interface Open {
  anchor: HTMLElement;
  source: SourceRef;
  mode: Mode;
}

interface PopoverApi {
  openFor: (anchor: HTMLElement, source: SourceRef, mode: Mode) => void;
  toggle: (anchor: HTMLElement, source: SourceRef) => void;
  closeFor: (anchor: HTMLElement, modes?: Mode[], focusingTo?: Node | null) => void;
  hoverOut: (anchor: HTMLElement) => void;
  /** Id of the popover while it describes this anchor. */
  /** The popover's id while it describes the trigger with this element id. */
  describedBy: (triggerId: string) => string | undefined;
  popoverId: string;
  current: HTMLElement | null;
}

const PopoverContext = createContext<PopoverApi | null>(null);

const GAP = 6;
const EDGE = 8;
const HOVER_IN_MS = 120;
const HOVER_OUT_MS = 160;
const SCROLL_QUIET_MS = 300;

export function CitationPopoverProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState<Open | null>(null);
  const [pos, setPos] = useState<{ top: number; left: number; above: boolean } | null>(null);
  const popRef = useRef<HTMLDivElement>(null);
  const hideTimer = useRef<number | null>(null);
  const showTimer = useRef<number | null>(null);
  const pointerInside = useRef(false);
  const popoverId = useId();
  const titleId = useId();

  const clearTimers = () => {
    if (hideTimer.current !== null) window.clearTimeout(hideTimer.current);
    if (showTimer.current !== null) window.clearTimeout(showTimer.current);
    hideTimer.current = showTimer.current = null;
  };

  // Content that scrolls under a resting pointer (the transcript following a streaming answer) fires mouseenter;
  // that isn't the user pointing at a chip, so hover-opens right after a scroll are ignored.
  const lastScroll = useRef(0);
  useEffect(() => {
    const onScroll = () => (lastScroll.current = performance.now());
    window.addEventListener("scroll", onScroll, { capture: true, passive: true });
    return () => window.removeEventListener("scroll", onScroll, { capture: true });
  }, []);

  const openFor = useCallback((anchor: HTMLElement, source: SourceRef, mode: Mode) => {
    if (mode === "hover" && performance.now() - lastScroll.current < SCROLL_QUIET_MS) return;
    clearTimers();
    const apply = () =>
      setOpen((cur) => {
        if (mode === "hover" && performance.now() - lastScroll.current < SCROLL_QUIET_MS) return cur;
        if (cur && cur.anchor === anchor) {
          // Already open for this chip: focus outranks hover (so leaving with the mouse doesn't close it while it
          // has keyboard focus); a click-pinned popover stays pinned.
          return mode === "focus" && cur.mode === "hover" ? { ...cur, mode } : cur;
        }
        return { anchor, source, mode };
      });
    if (mode === "hover") showTimer.current = window.setTimeout(apply, HOVER_IN_MS);
    else apply();
  }, []);

  const toggle = useCallback((anchor: HTMLElement, source: SourceRef) => {
    clearTimers();
    setOpen((cur) => (cur && cur.anchor === anchor && cur.mode === "click" ? null : { anchor, source, mode: "click" }));
  }, []);

  const closeFor = useCallback((anchor: HTMLElement, modes?: Mode[]) => {
    clearTimers();
    setOpen((cur) => (cur && cur.anchor === anchor && (!modes || modes.includes(cur.mode)) ? null : cur));
  }, []);

  const hoverOut = useCallback((anchor: HTMLElement) => {
    if (showTimer.current !== null) window.clearTimeout(showTimer.current);
    showTimer.current = null;
    if (hideTimer.current !== null) window.clearTimeout(hideTimer.current);
    hideTimer.current = window.setTimeout(() => {
      setOpen((cur) => (cur && cur.anchor === anchor && cur.mode === "hover" ? null : cur));
    }, HOVER_OUT_MS);
  }, []);

  useEffect(() => () => clearTimers(), []);

  // Place it under the chip (or above when there's no room), and keep it there while things scroll.
  const place = useCallback(() => {
    const pop = popRef.current;
    if (!open || !pop) return;
    if (!open.anchor.isConnected) return setOpen(null);
    const r = open.anchor.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    // Scrolled out of its scrolling area (the transcript), or off screen: close rather than float over the header.
    const area = open.anchor.closest("[data-scroll-root]")?.getBoundingClientRect() ?? { top: 0, bottom: vh };
    if (r.bottom < area.top || r.top > area.bottom) return setOpen(null);
    const { offsetWidth: w, offsetHeight: h } = pop;
    const left = Math.min(Math.max(EDGE, r.left + r.width / 2 - 28), vw - w - EDGE);
    let top = r.bottom + GAP;
    let above = false;
    if (top + h > vh - EDGE && r.top - GAP - h > EDGE) {
      top = r.top - GAP - h;
      above = true;
    }
    setPos((p) => (p && p.top === Math.round(top) && p.left === Math.round(left) && p.above === above ? p : { top: Math.round(top), left: Math.round(left), above }));
  }, [open]);

  useLayoutEffect(() => {
    if (!open) {
      setPos(null);
      return;
    }
    place();
  }, [open, place]);

  useEffect(() => {
    if (!open) return;
    let raf = 0;
    const onMove = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(place);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        // Focus was in the popover (its link): back to the chip, so it isn't lost when the popover goes.
        if (popRef.current?.contains(document.activeElement)) open.anchor.focus({ preventScroll: true });
        setOpen(null);
      }
    };
    const onPointer = (e: PointerEvent) => {
      const t = e.target as Node;
      if (popRef.current?.contains(t) || open.anchor.contains(t)) return;
      setOpen(null);
    };
    window.addEventListener("scroll", onMove, true);
    window.addEventListener("resize", onMove);
    document.addEventListener("keydown", onKey, true);
    document.addEventListener("pointerdown", onPointer, true);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("scroll", onMove, true);
      window.removeEventListener("resize", onMove);
      document.removeEventListener("keydown", onKey, true);
      document.removeEventListener("pointerdown", onPointer, true);
    };
  }, [open, place]);

  const anchor = open?.anchor ?? null;
  const api = useMemo<PopoverApi>(
    () => ({
      openFor,
      toggle,
      closeFor: (a, modes, focusingTo) => {
        if (pointerInside.current) return; // selecting text in the popover blurs the chip; keep it open
        if (focusingTo && popRef.current?.contains(focusingTo)) return; // focus is going to the popover's link
        closeFor(a, modes);
      },
      hoverOut,
      describedBy: (triggerId) => (anchor?.id === triggerId ? popoverId : undefined),
      popoverId,
      current: anchor,
    }),
    [openFor, toggle, closeFor, hoverOut, anchor, popoverId],
  );

  const source = open?.source;
  const pages = source ? pagesLong(source) : null;
  const web = source?.kind === "web";

  return (
    <PopoverContext value={api}>
      {children}
      {open &&
        source &&
        createPortal(
          <div
            ref={popRef}
            id={popoverId}
            role={source.url ? "dialog" : "tooltip"}
            aria-labelledby={titleId}
            className="cite-pop"
            data-kind={web ? "web" : undefined}
            data-above={pos?.above || undefined}
            style={pos ? { top: pos.top, left: pos.left } : { top: 0, left: 0, visibility: "hidden" }}
            onMouseEnter={() => {
              if (hideTimer.current !== null) window.clearTimeout(hideTimer.current);
              hideTimer.current = null;
            }}
            onMouseLeave={() => open.mode === "hover" && hoverOut(open.anchor)}
            onPointerDown={() => {
              pointerInside.current = true;
            }}
            onPointerUp={() => {
              window.setTimeout(() => (pointerInside.current = false), 0);
            }}
            onBlur={(e) => {
              // Focus left the popover (its link) for somewhere other than the chip: it has done its job.
              const to = e.relatedTarget as Node | null;
              if (open.mode !== "hover" && !pointerInside.current && !popRef.current?.contains(to) && to !== open.anchor) setOpen(null);
            }}
          >
            {web ? (
              <WebSourceCard source={source} titleId={titleId} anchor={open.anchor} />
            ) : (
              <>
                <div className="cite-pop-head">
                  {source.number !== null && <span className="cite-pop-num">{source.number}</span>}
                  <Icon name="doc" size={14} className="cite-pop-icon" />
                  <strong id={titleId} className="cite-pop-name">
                    {source.filename}
                  </strong>
                </div>
                {pages ? (
                  <p className="cite-pop-pages">{pages}</p>
                ) : (
                  sectionShort(source) && (
                    <p className="cite-pop-pages cite-pop-section" title={sectionPath(source) ?? undefined}>
                      {sectionShort(source)}
                    </p>
                  )
                )}
                {source.snippet ? (
                  <SnippetView snippet={source.snippet} />
                ) : (
                  <p className="cite-pop-empty">{source.note ?? "No passage was saved with this citation."}</p>
                )}
              </>
            )}
          </div>,
          document.body,
        )}
    </PopoverContext>
  );
}

/** A live web result in the popover: title, site and date, the result's text, and a link to the page. */
function WebSourceCard({ source, titleId, anchor }: { source: SourceRef; titleId: string; anchor: HTMLElement }) {
  const marker = markerOf(source);
  const published = publishedText(source.published);
  const site = source.site ?? source.filename;
  const host = hostOf(source.url);
  const title = source.title ?? site;
  const meta = [source.title ? site : null, published].filter(Boolean).join(" · ");
  // The popover sits at the end of the page, outside the chip's tab order: Tab from the link goes back to the chip
  // (and on from there), Shift+Tab to the chip.
  const onLinkKey = (e: ReactKeyboardEvent<HTMLAnchorElement>) => {
    if (e.key !== "Tab") return;
    if (e.shiftKey) e.preventDefault();
    anchor.focus({ preventScroll: true });
  };
  return (
    <>
      <div className="cite-pop-head">
        {marker && <span className="cite-pop-num cite-pop-num-web">{marker}</span>}
        <Icon name="globe" size={14} className="cite-pop-icon" />
        <strong id={titleId} className="cite-pop-name cite-pop-title">
          {title}
        </strong>
      </div>
      {meta && <p className="cite-pop-pages">{meta}</p>}
      {source.snippet ? (
        <blockquote className="cite-pop-snippet">{source.snippet}</blockquote>
      ) : (
        <p className="cite-pop-empty">{source.note ?? "No text was saved with this result."}</p>
      )}
      {source.url && (
        <a
          className="cite-pop-link"
          href={source.url}
          target="_blank"
          rel="noopener noreferrer"
          title={source.url}
          aria-label={`Open ${title}${host ? ` on ${host}` : ""} (opens in a new tab)`}
          onKeyDown={onLinkKey}
        >
          <Icon name="external" size={14} />
          <span>{host ? `Open on ${host}` : "Open the page"}</span>
        </a>
      )}
    </>
  );
}

/** The cited passage: a compact table when the chunk is a markdown table, quoted plain text otherwise. */
function SnippetView({ snippet }: { snippet: string }) {
  const table = useMemo(() => parseTableSnippet(snippet), [snippet]);
  // A column lines up to the right when every filled cell in it is a number (the first column labels the rows).
  const numeric = useMemo(
    () =>
      (table?.header ?? []).map((_, i) => {
        const cells = (table?.rows ?? []).map((r) => r[i] ?? "").filter((c) => c.trim() && c.trim() !== "-");
        return i > 0 && cells.length > 0 && cells.every(isNumericCell);
      }),
    [table],
  );
  if (!table) return <blockquote className="cite-pop-snippet">{snippet}</blockquote>;
  return (
    <div className="cite-pop-snippet is-table">
      {table.lead && <p className="cite-pop-lead">{table.lead}</p>}
      <div className="cite-pop-table-scroll">
        <table className="cite-pop-table" aria-label="Table excerpt">
          <thead>
            <tr>
              {table.header.map((h, i) => (
                <th key={i} scope="col" className={numeric[i] ? "num" : undefined}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, r) => (
              <tr key={r}>
                {row.map((cell, i) => (
                  <td key={i} className={numeric[i] ? "num" : undefined}>
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {table.truncated && <p className="cite-pop-more">More rows in the document…</p>}
    </div>
  );
}

/** The shared popover, for marks that are not chips (the canvas opens it from its data points). */
export function usePopover(): PopoverApi {
  const ctx = useContext(PopoverContext);
  if (!ctx) throw new Error("citation chips must be inside <CitationPopoverProvider>");
  return ctx;
}

/** Event handlers shared by inline chips and the source list buttons. */
function useSourceTrigger(source: SourceRef) {
  const pop = usePopover();
  const ref = useRef<HTMLButtonElement>(null);
  const id = useId();
  return {
    ref,
    id,
    "aria-describedby": pop.describedBy(id),
    onClick: () => ref.current && pop.toggle(ref.current, source),
    onFocus: () => {
      const el = ref.current;
      if (el && el.matches(":focus-visible")) pop.openFor(el, source, "focus");
    },
    onBlur: (e: ReactFocusEvent<HTMLButtonElement>) =>
      ref.current && pop.closeFor(ref.current, ["focus", "click"], e.relatedTarget as Node | null),
    // A web result's popover holds a link and sits at the end of the page: Tab from its open chip goes into the link.
    onKeyDown: (e: ReactKeyboardEvent<HTMLButtonElement>) => {
      const el = ref.current;
      if (e.key !== "Tab" || e.shiftKey || !el || !source.url || pop.current !== el) return;
      const link = document.getElementById(pop.popoverId)?.querySelector<HTMLElement>("a[href]");
      if (!link) return;
      e.preventDefault();
      link.focus({ preventScroll: true });
    },
    // Real mouse movement only: content scrolling under a resting pointer fires mouseenter but no pointermove.
    onPointerMove: (e: ReactPointerEvent<HTMLButtonElement>) => {
      const el = ref.current;
      if (e.pointerType === "mouse" && el && pop.current !== el) pop.openFor(el, source, "hover");
    },
    onMouseLeave: () => ref.current && pop.hoverOut(ref.current),
  };
}

/** One inline marker chip: "1" for `[S1]`, "W1" for `[W1]`. Unknown source ids stay visible but inert. */
export function CiteChip({ id, source }: { id: string; source: SourceRef | undefined }) {
  if (!source) {
    const web = /^W/i.test(id);
    return (
      <span className={`cite-chip missing${web ? " web" : ""}`} title={`${id}: this source isn't available`}>
        {web ? id.toUpperCase() : id.replace(/^S/i, "")}
      </span>
    );
  }
  return <CiteChipButton source={source} />;
}

function CiteChipButton({ source }: { source: SourceRef }) {
  const trigger = useSourceTrigger(source);
  return (
    <button
      type="button"
      className={source.kind === "web" ? "cite-chip web" : "cite-chip"}
      aria-label={describeSource(source)}
      {...trigger}
    >
      {markerOf(source) ?? "•"}
    </button>
  );
}

/**
 * Answer text with `[S#]` markers as chips. While `streaming`, a marker still being typed is held back and a caret
 * shows where text is arriving.
 */
export function AnswerText({
  text,
  sources,
  streaming = false,
}: {
  text: string;
  sources: Record<string, SourceRef>;
  streaming?: boolean;
}) {
  const parts = splitCitations(streaming ? withoutPartialMarker(text) : text);
  return (
    <>
      {parts.map((p, i) =>
        p.kind === "text" ? (
          <span key={i}>{p.text}</span>
        ) : (
          <span key={i} className="cite-group">
            {p.ids.map((id) => (
              <CiteChip key={id} id={id} source={sources[id]} />
            ))}
          </span>
        ),
      )}
      {streaming && <span className="caret" aria-hidden />}
    </>
  );
}

/** The sources a message cites, under the bubble; same numbers as the inline chips. Documents first, then web results. */
export function SourceList({ sources }: { sources: SourceRef[] }) {
  if (sources.length === 0) return null;
  const sorted = sortSources(sources);
  return (
    <ul className="cites" aria-label="Sources">
      {sorted.map((s) => (
        <li key={s.key}>
          <SourceButton source={s} />
        </li>
      ))}
    </ul>
  );
}

function SourceButton({ source }: { source: SourceRef }) {
  const trigger = useSourceTrigger(source);
  const where = locationShort(source);
  if (source.kind === "web") {
    // A live web result: a globe and the site (the page itself is one click further, in the popover).
    const marker = markerOf(source);
    return (
      <button type="button" className="cite cite-web" aria-label={describeSource(source)} {...trigger}>
        <Icon name="globe" size={13} />
        <span className="cite-name">{source.site ?? source.filename}</span>
        {marker && <span className="cite-page">{marker}</span>}
      </button>
    );
  }
  return (
    <button type="button" className="cite" aria-label={describeSource(source)} {...trigger}>
      {source.number !== null ? (
        <span className="cite-num" aria-hidden>
          {source.number}
        </span>
      ) : (
        <Icon name="doc" size={12} />
      )}
      <span className="cite-name">{source.filename}</span>
      {where &&
        (where.kind === "pages" ? (
          <span className="cite-page">{where.label}</span>
        ) : (
          <span className="cite-page cite-section" title={where.path ?? undefined}>
            {where.label}
          </span>
        ))}
    </button>
  );
}
