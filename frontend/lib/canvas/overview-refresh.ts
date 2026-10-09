/**
 * When the project's overview is read again, and what the page holds in between. Pure (no React, no fetch), so the
 * polling and the merge are checked in Node: `useOverview` (lib/canvas/use-canvas.ts) is the thin hook around them.
 *
 * The overview is built by the backend after a document is ingested (`GET /api/projects/{id}/overview`: `building`
 * while it works, then `ready` with the panels, or `none`). A page that stays open has to notice:
 *
 *   - the count of READY documents changed (an upload finished, a document was deleted): read again, and once more
 *     to confirm, because the build may start a moment after the document turns READY;
 *   - the status is `building`: read again with a growing pause, until it isn't (or a long time has passed);
 *   - `ready` (or `none`) and the answer is the same as the one before: stop.
 *
 * Nothing flickers: the panels last shown stay on screen while the overview is rebuilt, and a read that brings nothing
 * new returns the very same state object, so React draws nothing.
 */

import type { OverviewStatus } from "./client";

export const READ_BACKOFF_MS: readonly number[] = [1500, 2000, 3000, 4500, 6500, 10_000];
/** After a count changed and the answer was not `building`: the pause before reading again to confirm it. */
export const CONFIRM_MS = 1500;
/** A cycle (one mount, or one change of the count) stops reading after this long, whatever the backend says. */
export const READ_LIMIT_MS = 10 * 60_000;
/** Reads that fail (after one that worked) are tried again this many times before the page keeps what it has. */
export const MAX_FAILURES = 3;

/** The count of READY documents; null while the list is not loaded (so an empty project is not "unknown"). */
export function readyDocumentCount(docs: ReadonlyArray<{ status: string }> | null | undefined): number | null {
  return docs ? docs.filter((d) => d.status === "READY").length : null;
}

/**
 * Follows the count of READY documents. Each change (from one number to another) is a new `generation`; the first
 * number after the list loads (null → n) is not a change: the read on mount covers it.
 */
export interface CountWatch {
  projectId: string;
  count: number | null;
  generation: number;
}

export function watchCount(prev: CountWatch, projectId: string, count: number | null): CountWatch {
  if (prev.projectId !== projectId) return { projectId, count, generation: 0 };
  if (count === null || count === prev.count) return prev;
  return { projectId, count, generation: prev.count === null ? prev.generation : prev.generation + 1 };
}

// ------------------------------------------------------------------ what the page holds

interface Stamped {
  id: string;
  updated_at: string;
}

/** What one read of the overview said, with panels in the page's own type. */
export interface Read<P> {
  status: OverviewStatus;
  panels: P[];
}

/** What the page holds: the status last read and the panels last shown (kept through `building`). */
export interface Held<P> {
  projectId: string;
  status: OverviewStatus;
  panels: P[];
  /** Status and the panels' ids and versions: two reads with the same signature say the same thing. */
  signature: string;
}

export function signatureOf(status: OverviewStatus, panels: readonly Stamped[]): string {
  return `${status}|${panels.map((p) => `${p.id}@${p.updated_at}`).join(",")}`;
}

/**
 * The state after a read. `building` keeps the panels shown so far (the dashboard stays while it is rebuilt);
 * `ready` brings the new ones; `none` clears them. A read that changes nothing returns `prev` itself.
 */
export function mergeRead<P extends Stamped>(prev: Held<P> | null, projectId: string, read: Read<P>): Held<P> {
  const before = prev && prev.projectId === projectId ? prev : null;
  const panels = read.status === "ready" ? read.panels : read.status === "building" ? (before?.panels ?? []) : [];
  const signature = signatureOf(read.status, panels);
  if (before && before.signature === signature) return before;
  return { projectId, status: read.status, panels, signature };
}

/** A read that failed: with nothing held it is "no overview" (an older backend, or none reachable); else what the page has stays. */
export function failedRead<P extends Stamped>(prev: Held<P> | null, projectId: string): Held<P> {
  if (prev && prev.projectId === projectId) return prev;
  return { projectId, status: "none", panels: [], signature: signatureOf("none", []) };
}

// ------------------------------------------------------------------ when to read again

/**
 * Where a cycle of reads stands: how many came back, how many in a row were not `building`, and how many in a row said
 * the same thing (and were not `building`).
 */
export interface Cycle {
  reads: number;
  settling: number;
  stable: number;
  signature: string | null;
}

export const START_CYCLE: Cycle = { reads: 0, settling: 0, stable: 0, signature: null };

export function afterRead(cycle: Cycle, signature: string, status: OverviewStatus): Cycle {
  const building = status === "building";
  const same = !building && cycle.signature === signature;
  return {
    reads: cycle.reads + 1,
    settling: building ? 0 : cycle.settling + 1,
    stable: same ? cycle.stable + 1 : 0,
    signature,
  };
}

/** Reads in a row that are not `building`, after a count changed, before giving up on the answer settling. */
const SETTLE_READS = 4;

/**
 * The pause before the next read, or null to stop.
 *
 * - `building`: read again, the pause growing from 1.5 s to 10 s;
 * - otherwise a cycle that began at rest (the page opened) is done; one that began with a change of the count reads
 *   until two reads in a row say the same thing (the backend may start building a moment after the document turns
 *   READY, so the first answer can be the old one);
 * - and after READ_LIMIT_MS, never.
 */
export function nextDelay(cycle: Cycle, status: OverviewStatus, afterChange: boolean, elapsedMs: number): number | null {
  if (elapsedMs >= READ_LIMIT_MS) return null;
  if (status === "building") return READ_BACKOFF_MS[Math.min(cycle.reads - 1, READ_BACKOFF_MS.length - 1)] ?? null;
  if (!afterChange || cycle.stable >= 1 || cycle.settling >= SETTLE_READS) return null;
  return CONFIRM_MS;
}

/** The pause before trying again after a failed read (null: enough tries; the page keeps what it has). */
export function retryDelay(failures: number, elapsedMs: number): number | null {
  if (failures > MAX_FAILURES || elapsedMs >= READ_LIMIT_MS) return null;
  return READ_BACKOFF_MS[Math.min(failures - 1, READ_BACKOFF_MS.length - 1)] ?? null;
}
