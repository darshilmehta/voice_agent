/**
 * The canvas as the page holds it: the panels the server has, plus visuals still being prepared or that failed.
 * A pure reducer, so the sequences the backend streams (preparing → ready, preparing → failed, a full `canvas`
 * snapshot) are checked in Node.
 */

import { sortPanels } from "./order";
import type { CanvasEvent, Visual } from "./types";

export interface Pending {
  id: string;
  /** `preparing` shows a skeleton (or "Updating…" on a panel that exists); `failed` a quiet note. */
  phase: "preparing" | "failed";
  detail: string | null;
  /** Date.now() when it got into this phase, for expiry. */
  since: number;
}

export interface CanvasState {
  panels: Visual[];
  pending: Pending[];
  /** The canvas has been read from the server at least once (or an event gave one). */
  loaded: boolean;
  /** Bumped when a `ready` event named a visual without carrying it: the page fetches the canvas again. */
  refresh: number;
}

export const INITIAL_CANVAS: CanvasState = { panels: [], pending: [], loaded: false, refresh: 0 };

/** A visual that never became ready or failed (a lost event) stops showing its skeleton after this long. */
export const PREPARING_TTL_MS = 90_000;
/** A failure note is quiet and goes away by itself. */
export const FAILED_TTL_MS = 15_000;

export type CanvasAction =
  | { type: "snapshot"; panels: Visual[] }
  | { type: "event"; event: CanvasEvent; now: number }
  | { type: "dismiss"; id: string }
  | { type: "expire"; now: number };

const withoutId = (list: Pending[], id: string) => list.filter((p) => p.id !== id);

/** A snapshot ends the skeleton of every visual it brings onto the canvas; other pending visuals carry on. */
function settle(pending: Pending[], before: Visual[], after: Visual[]): Pending[] {
  const had = new Set(before.map((p) => p.id));
  const has = new Set(after.map((p) => p.id));
  return pending.filter((p) => !(p.phase === "preparing" && has.has(p.id) && !had.has(p.id)));
}

export function canvasReducer(state: CanvasState, action: CanvasAction): CanvasState {
  switch (action.type) {
    case "snapshot": {
      const panels = sortPanels(action.panels);
      return { ...state, panels, loaded: true, pending: settle(state.pending, state.panels, panels) };
    }
    case "event": {
      const ev = action.event;
      if (ev.type === "canvas") {
        const panels = sortPanels(ev.panels);
        return { ...state, panels, loaded: true, pending: settle(state.pending, state.panels, panels) };
      }
      if (ev.phase === "failed" && ev.detail === "cancelled") {
        // Given up because the conversation moved on (a newer question, stop): the skeleton just goes, no note.
        return { ...state, pending: withoutId(state.pending, ev.visualId) };
      }
      if (ev.phase === "preparing" || ev.phase === "failed") {
        const entry: Pending = { id: ev.visualId, phase: ev.phase, detail: ev.detail, since: action.now };
        return { ...state, pending: [...withoutId(state.pending, ev.visualId), entry] };
      }
      const pending = withoutId(state.pending, ev.visualId);
      if (!ev.visual) return { ...state, pending, refresh: state.refresh + 1 };
      const visual = ev.visual;
      return { ...state, panels: sortPanels([...state.panels.filter((p) => p.id !== visual.id), visual]), pending, loaded: true };
    }
    case "dismiss":
      return { ...state, pending: withoutId(state.pending, action.id) };
    case "expire": {
      const pending = state.pending.filter((p) => action.now - p.since < (p.phase === "preparing" ? PREPARING_TTL_MS : FAILED_TTL_MS));
      return pending.length === state.pending.length ? state : { ...state, pending };
    }
  }
}

/** Something to show: a panel, or a visual on its way (a skeleton). A lone failure note alone shows nothing. */
export function hasCanvasContent(state: CanvasState): boolean {
  return state.panels.length > 0 || state.pending.length > 0;
}
