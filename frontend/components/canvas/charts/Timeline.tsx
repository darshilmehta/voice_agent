"use client";

/**
 * Timeline (dates and obligations): a vertical list on a rail, which reads the same at 375 px and 1440 px. Dates are
 * written in the visual's language ("15 Mar 2025" / "15 मार्च 2025"); events with a free label ("On termination") keep
 * it as written. The order is chronological when every date is a date, otherwise the order the backend sent. Position
 * on the rail, the date text and the source beside each event carry the information, not colour.
 */

import { formatDate, isIsoDate, isoToTime } from "@/lib/canvas/format";
import type { TimelineEvent, Visual } from "@/lib/canvas/types";

import { CellButton, usePanel } from "../panel-context";

const norm = (s: string) => s.trim().toLowerCase();

export function orderedEvents(events: TimelineEvent[]): TimelineEvent[] {
  if (events.length === 0 || events.some((e) => !isIsoDate(e.date))) return events;
  return events
    .map((e, i) => ({ e, i }))
    .sort((a, b) => (isoToTime(a.e.date) ?? 0) - (isoToTime(b.e.date) ?? 0) || a.i - b.i)
    .map(({ e }) => e);
}

export function Timeline({ visual }: { visual: Visual }) {
  const { labels, resolve } = usePanel();
  const lang = visual.language;
  const hl = new Set((visual.highlight?.x ?? []).map(norm));
  const events = orderedEvents(visual.events);
  return (
    <ol className="cv-timeline" aria-label={visual.title}>
      {events.map((e, i) => {
        const strong = hl.has(norm(e.date)) || hl.has(norm(e.label));
        const iso = isIsoDate(e.date);
        const src = e.cell ? resolve(e.cell) : null;
        return (
          <li key={i} className="cv-event" data-highlight={strong || undefined}>
            <div className="cv-event-date">{iso ? <time dateTime={e.date}>{formatDate(e.date, lang)}</time> : <span>{e.date}</span>}</div>
            <span className="cv-rail" aria-hidden="true">
              <span className="cv-pt" />
            </span>
            <div className="cv-event-body">
              <p className="cv-event-label">
                {e.label}
                {strong && <span className="visually-hidden">, {labels.highlighted}</span>}
              </p>
              {e.detail && <p className="cv-event-detail">{e.detail}</p>}
              {e.cell && src && (
                <CellButton cell={e.cell} className="cv-event-src">
                  {src.filename}
                  {src.page !== null ? ` · ${labels.page} ${src.page}` : ""}
                </CellButton>
              )}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
