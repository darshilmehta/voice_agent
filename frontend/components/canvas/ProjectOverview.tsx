"use client";

/**
 * The project's overview dashboard, built from the tables found in its documents while they were ingested
 * (`GET /api/projects/{id}/overview`). Ready: its panels (read-only, each with "View as table" and its sources). Building:
 * a quiet "Building the overview…" while the backend works (re-read every few seconds). None, still loading, or an
 * endpoint the backend doesn't have: nothing at all.
 *
 * There is no "Show in chat" here: contract v1 has no way to put an existing visual on a chat's canvas (the canvas
 * operations are only remove, pin, unpin and move, and `POST …/visuals` is debug-only and takes a spec, not a visual),
 * so the button would have to fake it. Ask the question in a chat and the visual is built there.
 */

import { lazy, Suspense, useId } from "react";

import type { ProjectDocument } from "@/lib/api";
import { shellLabelsFor } from "@/lib/canvas/labels";
import { useOverview } from "@/lib/canvas/use-canvas";

import { CanvasBoundary } from "./Boundary";
import { DelayedSkeleton, PanelSkeleton } from "./CanvasPanel";

const LazyBoard = lazy(() => import("./CanvasBoard"));
const NONE: ReadonlySet<string> = new Set();

export function ProjectOverview({ projectId, docsById }: { projectId: string; docsById: Record<string, ProjectDocument> }) {
  const overview = useOverview(projectId);
  const titleId = useId();
  const labels = shellLabelsFor("en");
  if (overview.status === "loading" || overview.status === "none") return null;
  if (overview.status === "ready" && overview.panels.length === 0) return null;

  return (
    <section className="card cv-overview" aria-labelledby={titleId}>
      <div className="card-head">
        <h2 id={titleId}>{labels.overview}</h2>
      </div>
      {overview.status === "building" ? (
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
                <LazyBoard panels={overview.panels} updating={NONE} busy={NONE} docsById={docsById} readOnly />
              </Suspense>
            </CanvasBoundary>
          </div>
        </>
      )}
    </section>
  );
}
