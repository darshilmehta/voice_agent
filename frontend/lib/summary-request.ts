"use client";

/**
 * "Summarise" in a chat's menu (the sidebar, a project's chat list, the chat page's own header) asks the chat page to
 * open its Summary view and write the summary. The menu and the page don't share a parent, and the page may not be
 * mounted yet (the menu belongs to another chat and navigates first), so the request waits here, like
 * lib/voice/autostart.ts: the page takes it when it mounts, or at once when it is already open.
 */

import { useEffect } from "react";

const pending = new Set<string>();
const listeners = new Set<() => void>();

export function requestSummary(chatId: string): void {
  pending.add(chatId);
  for (const l of [...listeners]) l();
}

/** True once per request. */
export const takeSummaryRequest = (chatId: string): boolean => pending.delete(chatId);

/** Calls `onRequest` for each summary request for this chat, including one made before the page mounted. */
export function useSummaryRequests(chatId: string, onRequest: () => void): void {
  useEffect(() => {
    const check = () => {
      if (takeSummaryRequest(chatId)) onRequest();
    };
    check();
    listeners.add(check);
    return () => void listeners.delete(check);
  }, [chatId, onRequest]);
}
