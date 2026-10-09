"use client";

/**
 * Automatic titles (docs/DESIGN.md §3.9): a chat is created as "New chat", and shortly after its first answer is saved
 * the backend gives it a title from the question. No event says so, so while the title is still the placeholder (and
 * nobody has renamed the chat) the page asks again: 0.8, 1.8, 3.2 and 5 s after each saved answer, then less and less
 * often for about a minute (the model can be busy), and once more whenever the user comes back to the tab or window.
 * It stops the moment the title changes or the user renames the chat, and when the page goes away.
 * The timing lives in `lib/title-watch.ts`.
 *
 * The chat is refetched into the shared store without a loading state, so the header and the sidebar change their
 * text in place (fading it in, components/TitleText.tsx) instead of flickering.
 */

import { useEffect, useRef } from "react";

import { PLACEHOLDER_TITLE, type Chat } from "./api";
import { watchTitle, type TitleWatchEnv } from "./title-watch";
import { useWorkspaceActions } from "./workspace";

/** The chat still carries the placeholder title the backend is expected to replace. */
export const awaitsAutoTitle = (chat: Pick<Chat, "title" | "title_is_auto">): boolean =>
  chat.title_is_auto && chat.title === PLACEHOLDER_TITLE;

/** The browser behind `watchTitle`: window timers, the tab's visibility, and "came back" = focus or visible again. */
function browserEnv(refresh: () => Promise<unknown>): TitleWatchEnv {
  return {
    refresh,
    setTimeout: (fn, ms) => window.setTimeout(fn, ms),
    clearTimeout: (handle) => window.clearTimeout(handle as number),
    now: () => Date.now(),
    visible: () => document.visibilityState === "visible",
    onReturn: (listener) => {
      window.addEventListener("focus", listener);
      document.addEventListener("visibilitychange", listener);
      return () => {
        window.removeEventListener("focus", listener);
        document.removeEventListener("visibilitychange", listener);
      };
    },
  };
}

/** A question and its answer: a chat with at least this many messages has had an answer that may still await a title. */
const ANSWERED_MESSAGES = 2;

/**
 * Re-reads the chat while it awaits its title. `answers` is how many answers this page has seen saved (voice
 * `agent_message`s plus completed typed turns): each increase starts a fresh minute of checks. A chat that already has
 * an answer when the page opens is watched too (a reload, or a return to the chat shortly after the answer); one with
 * no answer yet is not, since no title can come before it.
 */
export function useAutoTitleRefresh(chat: Chat, answers: number): void {
  const ws = useWorkspaceActions();
  const chatId = chat.id;
  const projectId = chat.project_id;
  const waiting = awaitsAutoTitle(chat);
  const active = waiting && (answers > 0 || chat.message_count >= ANSWERED_MESSAGES);

  useEffect(() => {
    if (!active) return;
    // Leaving the page, another chat, a rename or the title arriving (active turns false) cancels what is left.
    return watchTitle(browserEnv(() => ws.loadChat(chatId, true)));
  }, [active, answers, chatId, ws]);

  // The title arrived (not a rename: that refreshes the list itself): the write also changes the chat's update time,
  // so read the sidebar's list once, rather than on every check.
  const wasActive = useRef(false);
  useEffect(() => {
    if (wasActive.current && !waiting && chat.title_is_auto) void ws.loadChats(projectId, true);
    wasActive.current = active;
  }, [active, waiting, chat.title_is_auto, projectId, ws]);
}
