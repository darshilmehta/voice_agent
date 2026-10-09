"use client";

/**
 * The project's overview dashboard, built from the tables found in its documents while they were ingested
 * (`GET /api/projects/{id}/overview`). Ready: its panels (read-only, each with "View as table" and its sources). Building:
 * a quiet "Building the overview…" while the backend works, or, when panels are already shown (a document was added or
 * removed), they stay and say "Updating…" until the new ones arrive. None, still loading, or an endpoint the backend
 * doesn't have: nothing at all. The overview follows the project while the page stays open: it is read again when the
 * count of READY documents changes and while it is being built (`useOverview`).
 *
 * There is no "Show in chat" here: contract v1 has no way to put an existing visual on a chat's canvas (the canvas
 * operations are only remove, pin, unpin and move, and `POST …/visuals` is debug-only and takes a spec, not a visual),
 * so the button would have to fake it. Ask the question in a chat and the visual is built there.
 */

import { lazy, Suspense, useId, useMemo } from "react";

import type { ProjectDocument } from "@/lib/api";
import { shellLabelsFor } from "@/lib/canvas/labels";
import { useOverview } from "@/lib/canvas/use-canvas";

import { CanvasBoundary } from "./Boundary";
import { DelayedSkeleton, PanelSkeleton } from "./CanvasPanel";

const LazyBoard = lazy(() => import("./CanvasBoard"));
const NONE: ReadonlySet<string> = new Set();

export function ProjectOverview({
  projectId,
  docsById,
  readyDocuments,
}: {
  projectId: string;
  docsById: Record<string, ProjectDocument>;
  /** How many of the project's documents are READY (null until they are listed): a change reads the overview again. */
  readyDocuments: number | null;
}) {
  const overview = useOverview(projectId, readyDocuments);
  const titleId = useId();
  const labels = shellLabelsFor("en");
  // While it is rebuilt the panels last shown stay, each saying "Updating…" (no skeleton in their place: no flicker).
  const updating = useMemo(
    () => (overview.status === "building" ? new Set(overview.panels.map((p) => p.id)) : NONE),
    [overview.status, overview.panels],
  );
  if (overview.status === "loading" || overview.status === "none") return null;
  if (overview.status === "ready" && overview.panels.length === 0) return null;

  return (
    <section className="card cv-overview" aria-labelledby={titleId}>
      <div className="card-head">
        <h2 id={titleId}>{labels.overview}</h2>
      </div>
      {overview.status === "building" && overview.panels.length === 0 ? (
        <div className="cv-board cv-overview-grid">
          <PanelSkeleton text={labels.overviewBuilding} />
        </div>
      ) : (
        <>
          <p className="sub cv-overview-sub">Key figures and trends found in this project&apos;s documents. A figure opens the cell it came from; a calculated one says so and shows how.</p>
          <div className="cv-board cv-overview-grid">
            <CanvasBoundary
              fallback={() => <p className="cv-note">{labels.failed}</p>}
            >
              <Suspense fallback={overview.panels.map((p) => <DelayedSkeleton key={p.id} text={labels.preparing} />)}>
                <LazyBoard panels={overview.panels} updating={updating} busy={NONE} docsById={docsById} readOnly />
              </Suspense>
            </CanvasBoundary>
          </div>
        </>
      )}
    </section>
  );
}
