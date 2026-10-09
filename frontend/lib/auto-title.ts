"use client";

/**
 * Automatic titles (docs/DESIGN.md §3.9): a chat is created as "New chat", and about a second or two after its first
 * answer is saved the backend gives it a title from the question. No event says so, so the page asks again a few
 * times after each saved answer, while the title is still the placeholder and nobody has renamed it, and stops as
 * soon as the title changes or after about five seconds.
 *
 * The chat is refetched into the shared store without a loading state, so the header and the sidebar change their
 * text in place (fading it in, components/TitleText.tsx) instead of flickering.
 */

import { useEffect } from "react";

import { PLACEHOLDER_TITLE, type Chat } from "./api";
import { useWorkspaceActions } from "./workspace";

/**
 * Delays after an answer was saved: the title lands within about 1 to 2 s, so look at 0.8 s, 1.8 s and 3.2 s, and once
 * more at 5 s for a slow model. The gaps widen (0.8, 1.0, 1.4, 1.8 s).
 */
export const TITLE_POLL_MS: readonly number[] = [800, 1800, 3200, 5000];

/** The chat still carries the placeholder title the backend is expected to replace. */
export const awaitsAutoTitle = (chat: Pick<Chat, "title" | "title_is_auto">): boolean =>
  chat.title_is_auto && chat.title === PLACEHOLDER_TITLE;

/**
 * Re-reads the chat after each saved answer while it awaits its title. `answers` is how many answers this page has
 * seen saved (voice `agent_message`s plus completed typed turns): each increase starts a fresh round of retries.
 */
export function useAutoTitleRefresh(chat: Chat, answers: number): void {
  const ws = useWorkspaceActions();
  const chatId = chat.id;
  const projectId = chat.project_id;
  const waiting = awaitsAutoTitle(chat);

  useEffect(() => {
    if (!waiting || answers === 0) return;
    const timers = TITLE_POLL_MS.map((delay) =>
      window.setTimeout(() => {
        void ws.loadChat(chatId, true);
        void ws.loadChats(projectId, true); // the new title also moves the chat in the sidebar's order
      }, delay),
    );
    // Leaving the page, or the title arriving (waiting turns false), cancels what is left.
    return () => timers.forEach((t) => window.clearTimeout(t));
  }, [waiting, answers, chatId, projectId, ws]);
}
