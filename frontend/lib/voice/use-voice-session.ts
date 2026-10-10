"use client";

import { useEffect, useState, useSyncExternalStore } from "react";

import type { Language } from "../api";
import { useBackend } from "../backend-context";
import { loadInputMode } from "./input-mode";
import { VoiceSession, type VoiceSnapshot } from "./session";

/**
 * The voice session of one chat, and its state. The session is created with the page and torn down when the page
 * goes (navigating away stops the microphone and the socket); it does nothing until `start()`.
 */
export function useVoiceSession(chatId: string, language: Language | null): { session: VoiceSession; snapshot: VoiceSnapshot } {
  const { backendUrl, config } = useBackend();
  // `features.web_search`: with it off the web search's badge, label and note stay off (docs/DESIGN.md §3.7).
  const webSearch = config?.features.web_search === true;
  // The denoiser and the noise-floor gate (§3.10); a backend without them leaves the microphone as it was.
  const voiceInput = config?.voice_input;
  const [session] = useState(() => new VoiceSession({ chatId, backendUrl, language, webSearch, voiceInput }));
  const snapshot = useSyncExternalStore(session.subscribe, session.getSnapshot, session.getSnapshot);

  useEffect(() => {
    session.setOptions({ language, backendUrl, webSearch, voiceInput });
  }, [session, language, backendUrl, webSearch, voiceInput]);

  useEffect(() => {
    session.setInputMode(loadInputMode()); // this viewer's choice of hold-to-talk, remembered in the browser
  }, [session]);

  useEffect(() => {
    // Development aid: window.__voice is the live session (counters, snapshot) for debugging and browser tests.
    if (process.env.NODE_ENV !== "production") (window as unknown as { __voice?: VoiceSession }).__voice = session;
    return () => session.destroy();
  }, [session]);

  return { session, snapshot };
}
