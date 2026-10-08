"use client";

import { useEffect, useState, useSyncExternalStore } from "react";

import type { Language } from "../api";
import { useBackend } from "../backend-context";
import { VoiceSession, type VoiceSnapshot } from "./session";

/**
 * The voice session of one chat, and its state. The session is created with the page and torn down when the page
 * goes (navigating away stops the microphone and the socket); it does nothing until `start()`.
 */
export function useVoiceSession(chatId: string, language: Language | null): { session: VoiceSession; snapshot: VoiceSnapshot } {
  const { backendUrl } = useBackend();
  const [session] = useState(() => new VoiceSession({ chatId, backendUrl, language }));
  const snapshot = useSyncExternalStore(session.subscribe, session.getSnapshot, session.getSnapshot);

  useEffect(() => {
    session.setOptions({ language, backendUrl });
  }, [session, language, backendUrl]);

  useEffect(() => {
    // Development aid: window.__voice is the live session (counters, snapshot) for debugging and browser tests.
    if (process.env.NODE_ENV !== "production") (window as unknown as { __voice?: VoiceSession }).__voice = session;
    return () => session.destroy();
  }, [session]);

  return { session, snapshot };
}
