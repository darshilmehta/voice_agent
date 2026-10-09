/**
 * Waiting for an automatic title (docs/DESIGN.md §3.9), without React or the DOM so it can be checked on its own.
 *
 * The backend writes the title in the background after the first answer. Nothing announces it, so the page asks again
 * while the chat still has the placeholder: quickly at first (the title usually lands within a second or two), then
 * less and less often, for about a minute (a busy model can take longer). Asking also happens when the user comes
 * back to the tab or window. It never asks while the tab is hidden, and `stop()` ends everything at once.
 */

/** How long after an answer the page keeps asking. */
export const TITLE_WINDOW_MS = 60_000;

/** The first checks, as delays after the answer: the title usually lands within about 1 to 2 s. */
const EARLY_MS: readonly number[] = [800, 1800, 3200, 5000];
const GAP_GROWTH = 1.4;
const MAX_GAP_MS = 10_000;

/**
 * Delays after an answer was saved at which to re-read the chat: 0.8, 1.8, 3.2 and 5 s, then the gap widens by 40%
 * (2.5, 3.5, 4.9 … s) up to 10 s, and the last check is at the end of the window. 12 reads in a minute.
 */
export function titleSchedule(windowMs: number = TITLE_WINDOW_MS): number[] {
  const delays = EARLY_MS.filter((d) => d < windowMs);
  let gap = Math.max(1000, (delays[delays.length - 1] ?? 0) - (delays[delays.length - 2] ?? 0));
  for (let last = delays[delays.length - 1] ?? 0; last < windowMs; ) {
    gap = Math.min(MAX_GAP_MS, Math.round((gap * GAP_GROWTH) / 100) * 100);
    last = Math.min(windowMs, last + gap);
    delays.push(last);
  }
  return delays;
}

export const TITLE_POLL_MS: readonly number[] = titleSchedule();

/** Two reads closer together than this are one (a timer and a focus event often land together). */
const MIN_GAP_MS = 500;

export interface TitleWatchEnv {
  /** Re-read the chat from the server into the shared store. */
  refresh: () => Promise<unknown> | void;
  setTimeout: (fn: () => void, ms: number) => unknown;
  clearTimeout: (handle: unknown) => void;
  now: () => number;
  /** False while the tab is in the background: nobody would see the title yet. */
  visible: () => boolean;
  /** Calls the listener when the user returns to the tab or window; returns the unsubscribe. */
  onReturn: (listener: () => void) => () => void;
}

/**
 * Start watching for the title: re-reads the chat on `titleSchedule()`, and whenever the user comes back. Returns
 * `stop`, which cancels the timers and the listeners (call it when the title arrives, the user renames the chat, the
 * page unmounts or another chat opens). After the schedule has run out the listeners stay, so coming back to a chat
 * that is still untitled looks once more.
 */
export function watchTitle(env: TitleWatchEnv, schedule: readonly number[] = TITLE_POLL_MS): () => void {
  let stopped = false;
  let lastRead = Number.NEGATIVE_INFINITY;

  const read = () => {
    if (stopped) return;
    const now = env.now();
    if (now - lastRead < MIN_GAP_MS) return;
    lastRead = now;
    try {
      void Promise.resolve(env.refresh()).catch(() => undefined);
    } catch {
      // a failed read is not fatal: the next check asks again
    }
  };
  const timers = schedule.map((delay) => env.setTimeout(() => env.visible() && read(), delay));
  const unsubscribe = env.onReturn(() => env.visible() && read());

  return () => {
    stopped = true;
    timers.forEach((t) => env.clearTimeout(t));
    unsubscribe();
  };
}
