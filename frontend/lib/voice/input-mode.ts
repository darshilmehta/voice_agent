/**
 * The viewer's choice between talking freely and hold-to-talk (docs/DESIGN.md §3.10 "Noisy rooms"), remembered in this
 * browser. Storage can be missing or throw (a private window, blocked site data): then the choice lasts for the page.
 */

import type { InputMode } from "./protocol";

const KEY = "voice.inputMode";

export function loadInputMode(): InputMode {
  try {
    return window.localStorage.getItem(KEY) === "ptt" ? "ptt" : "vad";
  } catch {
    return "vad";
  }
}

export function saveInputMode(mode: InputMode): void {
  try {
    window.localStorage.setItem(KEY, mode);
  } catch {
    // not remembered: fine
  }
}
