/**
 * The quiet trace of a live web search (docs/DESIGN.md §3.7): a small "Searching the web…" badge while it runs, then
 * "Searched the web for “…”" once it has (or "…· timed out", "Web search failed for “…”"). The query is shown because
 * it is exactly what left the machine. Used under the streaming answer, under a saved answer and on the voice stage;
 * the callers show it only when `features.web_search` is on.
 *
 * Plain text for screen readers: the badge is a visual companion of the state announcements ("Searching the web…" is
 * announced by the composer's status line or the voice state label), so it carries no live region of its own.
 */

import { searchDetail, searchNote, type WebSearchState } from "@/lib/web-search";

import { Icon } from "./Icon";

export function WebSearchNote({ search, stage = false }: { search: WebSearchState; stage?: boolean }) {
  const searching = search.status === "searching";
  const note = searchNote(search);
  return (
    <p
      className={`web-note${searching ? " web-badge" : ""}${stage ? " is-stage" : ""}`}
      data-state={search.status}
      title={searching ? undefined : searchDetail(search)}
    >
      <Icon name="globe" size={13} className="web-globe" />
      {searching ? (
        <>
          <span className="web-lead">Searching the web…</span>
          {search.query && " "}
          {search.query && <q className="web-query">{search.query}</q>}
        </>
      ) : (
        <>
          <span className="web-lead">{note.lead}</span>
          {note.query && " "}
          {note.query && <q className="web-query">{note.query}</q>}
          {note.tail && " "}
          {note.tail && <span className="web-tail">· {note.tail}</span>}
        </>
      )}
    </p>
  );
}
