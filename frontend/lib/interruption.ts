/**
 * How the transcript says a saved answer was cut short. A voice answer that was interrupted is saved with
 * `heard_text` (what was actually played) and `route.interrupted`: why it was cut.
 *
 *   barge_in    the user spoke over it
 *   stop        the user pressed Stop (or Esc)
 *   disconnect  the connection ended: the client dropped, another tab took over the chat, or the server shut down
 *
 * Older messages and text answers have no reason; they read as before.
 */

import type { Message } from "./api";

export type CutReason = "barge_in" | "stop" | "disconnect";

export function cutReasonOf(m: Message): CutReason | null {
  const reason = (m.route as Record<string, unknown> | null)?.interrupted;
  return reason === "barge_in" || reason === "stop" || reason === "disconnect" ? reason : null;
}

/** "Interrupted after: “…”", or without a quote when nothing was played (`heard` is the words that were). */
export function cutNote(reason: CutReason | null, heard: string): { lead: string; quote: string | null } {
  if (heard.trim()) {
    const lead =
      reason === "stop"
        ? "Stopped after:"
        : reason === "disconnect"
          ? "Cut off when the connection dropped, after:"
          : "Interrupted after:";
    return { lead, quote: heard };
  }
  const lead =
    reason === "stop"
      ? "Stopped before it was played"
      : reason === "disconnect"
        ? "Cut off before it was played, when the connection dropped"
        : "Interrupted before it was played";
  return { lead, quote: null };
}

/**
 * The quiet note under an answer that has no "after:" note but was cut: stopped while it was being written
 * (`stopped`, from `route.stopped`), so either nothing was written (`blank`) or the written part is all there is; or,
 * without that flag, cut by a dropped connection with nothing recorded about how much was heard.
 */
export function stoppedNote(reason: CutReason | null, blank: boolean, stopped = true): string {
  if (reason === "disconnect") {
    if (blank) return "Cut off before the answer started, when the connection dropped.";
    return stopped
      ? "Cut off when the connection dropped. The rest of this answer wasn't written."
      : "Cut off when the connection dropped.";
  }
  if (reason === "barge_in") {
    return blank ? "Interrupted before the answer started." : "Interrupted. The rest of this answer wasn't written.";
  }
  return blank ? "Stopped before the answer started." : "Stopped. The rest of this answer wasn't written.";
}
