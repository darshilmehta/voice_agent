/**
 * Where canvas events from the turn streams meet the canvas. Both the text stream (SSE `event: visual` / `event:
 * canvas`, read in lib/api.ts and passed on by lib/chat-turns.ts) and the voice socket (`{type: "visual" | "canvas"}`,
 * lib/voice/session.ts) hand what they read to `publishRawCanvasEvent`; the open chat page's canvas subscribes. An event
 * for a chat nobody is showing is dropped: the canvas is fetched whole when the page opens.
 *
 * The contract reader (./types) is loaded when the first event needs it, not with the page; events are read in the
 * order they arrived.
 */

import type { CanvasEvent } from "./types";

type Listener = (event: CanvasEvent) => void;

const listeners = new Map<string, Set<Listener>>();

export function subscribeCanvasEvents(chatId: string, listener: Listener): () => void {
  let set = listeners.get(chatId);
  if (!set) listeners.set(chatId, (set = new Set()));
  const mine = set;
  mine.add(listener);
  return () => {
    mine.delete(listener);
    if (mine.size === 0) listeners.delete(chatId);
  };
}

export function publishCanvasEvent(chatId: string, event: CanvasEvent): void {
  for (const l of [...(listeners.get(chatId) ?? [])]) l(event);
}

let queue: Promise<void> = Promise.resolve();

/** For a stream that hands over the event name and its parsed JSON; anything that isn't a canvas event is ignored. */
export function publishRawCanvasEvent(chatId: string, name: string, data: unknown): void {
  queue = queue
    .then(async () => {
      const event = (await import("./types")).parseCanvasEvent(name, data);
      if (event) publishCanvasEvent(chatId, event);
    })
    .catch(() => undefined); // a canvas that can't be read never breaks the conversation
}
