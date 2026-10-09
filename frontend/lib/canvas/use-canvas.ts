"use client";

/**
 * The canvas of one chat, and the overview of one project, as React state.
 *
 * `useCanvas` reads `GET /api/chats/{id}/canvas` when the chat opens, then follows the events the turn streams carry
 * (`lib/canvas/events.ts`): a visual preparing (skeleton), ready or failed, and full `canvas` snapshots. Pin, unpin,
 * remove and move go to `POST …/canvas/ops`; the answer is the new canvas, which replaces what the page holds.
 * Everything is optional: if the backend has no canvas (404), is unreachable, or has nothing yet, the page shows no
 * canvas at all and nothing else is affected.
 */

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";

import { errorMessage, isAbort } from "../api";
import { useBackend } from "../backend-context";
import { useToast } from "@/components/Toast";
import { createCanvasApi } from "./client";
import { subscribeCanvasEvents } from "./events";
import { shellLabelsFor } from "./labels";
import {
  START_CYCLE,
  afterRead,
  failedRead,
  mergeRead,
  nextDelay,
  retryDelay,
  signatureOf,
  watchCount,
  type CountWatch,
  type Held,
} from "./overview-refresh";
import { INITIAL_CANVAS, canvasReducer, hasCanvasContent, nextSkeletonIn, visiblePending, type Pending } from "./state";
import type { CanvasOp, Visual } from "./types";

export interface CanvasController {
  panels: Visual[];
  /** The visuals on their way worth showing: skeletons that have waited SKELETON_DELAY_MS, and failure notes. */
  pending: Pending[];
  /** A visual is on its way, shown yet or not (the chart code can start loading). */
  preparing: boolean;
  /** Something to show: a panel or a skeleton. False draws no canvas chrome at all. */
  hasContent: boolean;
  /** Ids with a pin / remove / move in flight. */
  busy: ReadonlySet<string>;
  /** For a polite live region: new visuals, removals and moves. */
  announcement: string;
  op: (op: CanvasOp, visualId: string, position?: number) => Promise<boolean>;
  dismiss: (id: string) => void;
  /** Read the canvas again (after the voice connection dropped and came back, events may have been missed). */
  reload: () => void;
}

const OP_VERB: Record<CanvasOp, string> = { remove: "remove the visual", pin: "pin the visual", unpin: "unpin the visual", move: "move the visual" };

export function useCanvas(chatId: string): CanvasController {
  const { backendUrl } = useBackend();
  const client = useMemo(() => createCanvasApi(backendUrl), [backendUrl]);
  const toast = useToast();
  const [state, dispatch] = useReducer(canvasReducer, INITIAL_CANVAS);
  const [busy, setBusy] = useState<ReadonlySet<string>>(() => new Set());
  const [announcement, setAnnouncement] = useState("");
  const [reloads, setReloads] = useState(0);
  const reload = useCallback(() => setReloads((n) => n + 1), []);

  // Read the canvas when the chat opens, and again when a `ready` event arrived without its visual (or on `reload`).
  const refresh = state.refresh;
  useEffect(() => {
    const ctrl = new AbortController();
    client
      .getCanvas(chatId, ctrl.signal)
      .then((panels) => dispatch({ type: "snapshot", panels }))
      .catch((err: unknown) => {
        if (isAbort(err) || ctrl.signal.aborted) return;
        // No canvas endpoint (older backend), or the backend is down: no canvas, no complaint. The page has its own
        // way of saying the backend is unreachable.
      });
    return () => ctrl.abort();
  }, [client, chatId, refresh, reloads]);

  // The turn streams (text SSE and the voice socket) publish here.
  useEffect(() => subscribeCanvasEvents(chatId, (event) => dispatch({ type: "event", event, now: Date.now() })), [chatId]);

  // A skeleton waits SKELETON_DELAY_MS for its `ready` before it shows (lib/canvas/state.ts): this clock is what it is
  // judged by, moved to the moment the next one is due. It starts at 0, so nothing shows before its own timer fires.
  const [clock, setClock] = useState(0);
  useEffect(() => {
    const wait = nextSkeletonIn(state.pending, Date.now());
    if (wait === null) return;
    const timer = window.setTimeout(() => setClock(Date.now()), wait);
    return () => window.clearTimeout(timer);
  }, [state.pending, clock]); // (a timer that fired a millisecond early finds the skeleton not yet due and waits again)
  const shown = useMemo(() => visiblePending(state.pending, clock), [state.pending, clock]);

  // A visual that never settles (a lost event) doesn't keep its skeleton forever.
  const hasPending = state.pending.length > 0;
  useEffect(() => {
    if (!hasPending) return;
    const timer = window.setInterval(() => dispatch({ type: "expire", now: Date.now() }), 4000);
    return () => window.clearInterval(timer);
  }, [hasPending]);

  // New visuals and removals are announced; the first read of the canvas is not.
  const known = useRef<Map<string, string> | null>(null);
  useEffect(() => {
    if (!state.loaded) return;
    const now = new Map(state.panels.map((p) => [p.id, p.title] as const));
    const before = known.current;
    known.current = now;
    if (!before) return;
    for (const p of state.panels) {
      if (!before.has(p.id)) {
        const l = shellLabelsFor(p.language);
        setAnnouncement(l.added(p.title, p.summary));
      }
    }
    for (const [id, title] of before) if (!now.has(id)) setAnnouncement(shellLabelsFor("en").removed(title));
  }, [state.loaded, state.panels]);

  const op = useCallback(
    async (kind: CanvasOp, visualId: string, position?: number): Promise<boolean> => {
      setBusy((s) => new Set(s).add(visualId));
      try {
        const panels = await client.canvasOp(chatId, { op: kind, visual_id: visualId, ...(position !== undefined ? { position } : {}) });
        dispatch({ type: "snapshot", panels });
        if (kind === "move") {
          const at = panels.findIndex((p) => p.id === visualId);
          const moved = panels[at];
          if (moved) setAnnouncement(shellLabelsFor(moved.language).moved(moved.title, at + 1, panels.length));
        }
        return true;
      } catch (err) {
        if (!isAbort(err)) toast({ tone: "error", message: `Couldn't ${OP_VERB[kind]}: ${errorMessage(err)}` });
        void client.getCanvas(chatId).then((panels) => dispatch({ type: "snapshot", panels })).catch(() => undefined);
        return false;
      } finally {
        setBusy((s) => {
          const next = new Set(s);
          next.delete(visualId);
          return next;
        });
      }
    },
    [client, chatId, toast],
  );

  const dismiss = useCallback((id: string) => dispatch({ type: "dismiss", id }), []);

  return {
    panels: state.panels,
    pending: shown,
    preparing: state.pending.some((p) => p.phase === "preparing"),
    hasContent: hasCanvasContent({ ...state, pending: shown }),
    busy,
    announcement,
    op,
    dismiss,
    reload,
  };
}

// ------------------------------------------------------------------ the project's overview

export type OverviewState =
  | { status: "loading" | "none"; panels: [] }
  /** Being (re)built: the panels last shown stay (none yet: the page shows a skeleton). */
  | { status: "building"; panels: Visual[] }
  | { status: "ready"; panels: Visual[] };

/**
 * `GET /api/projects/{id}/overview`, kept up to date while the page stays open (the rules are in
 * lib/canvas/overview-refresh.ts): read again, with a growing pause, while the backend says `building`; read again
 * when `readyDocuments` (the project's count of READY documents) changes, and once more to confirm it; stop when it
 * is `ready` and says the same twice. The panels last shown stay while it is rebuilt, and a read that brings nothing
 * new changes nothing, so nothing flickers.
 */
export function useOverview(projectId: string, readyDocuments: number | null = null): OverviewState {
  const { backendUrl } = useBackend();
  const client = useMemo(() => createCanvasApi(backendUrl), [backendUrl]);
  const [held, setHeld] = useState<Held<Visual> | null>(null);
  const heldRef = useRef(held);
  useEffect(() => {
    heldRef.current = held;
  }, [held]);

  // Each change of the count of READY documents starts a new run of reads (derived while rendering, as React allows).
  const [watch, setWatch] = useState<CountWatch>({ projectId, count: readyDocuments, generation: 0 });
  const seen = watchCount(watch, projectId, readyDocuments);
  if (seen !== watch) setWatch(seen);
  const generation = seen.generation;

  useEffect(() => {
    const ctrl = new AbortController();
    const afterChange = generation > 0;
    const startedAt = Date.now();
    let timer = 0;
    let cycle = START_CYCLE;
    let failures = 0;
    let parked = false; // a read is due, but the tab is hidden: it waits for the tab to come back

    const schedule = (ms: number) => {
      timer = window.setTimeout(due, ms);
    };
    const due = () => {
      if (document.visibilityState === "hidden") parked = true;
      else void read();
    };
    const onVisible = () => {
      if (parked && document.visibilityState === "visible") {
        parked = false;
        void read();
      }
    };
    const read = async () => {
      try {
        const value = await client.getOverview(projectId, ctrl.signal);
        if (ctrl.signal.aborted) return;
        failures = 0;
        setHeld((prev) => mergeRead(prev, projectId, value));
        cycle = afterRead(cycle, signatureOf(value.status, value.panels), value.status);
        const delay = nextDelay(cycle, value.status, afterChange, Date.now() - startedAt);
        if (delay !== null) schedule(delay);
      } catch (err) {
        if (isAbort(err) || ctrl.signal.aborted) return;
        const kept = heldRef.current;
        if (!kept || kept.projectId !== projectId) {
          // No overview endpoint, or the backend is down: nothing to show.
          setHeld((prev) => failedRead(prev, projectId));
          return;
        }
        failures += 1; // (what the page shows stays; a few more tries)
        const delay = retryDelay(failures, Date.now() - startedAt);
        if (delay !== null) schedule(delay);
      }
    };

    document.addEventListener("visibilitychange", onVisible);
    void read();
    return () => {
      ctrl.abort();
      window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [client, projectId, generation]);

  const value = held && held.projectId === projectId ? held : null;
  if (!value) return { status: "loading", panels: [] };
  if (value.status === "none") return { status: "none", panels: [] };
  return { status: value.status, panels: value.panels };
}
