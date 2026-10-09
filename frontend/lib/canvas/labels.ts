/**
 * The few words the canvas shell needs before any chart exists: the skeleton, a failed visual, the swipe indicator, the
 * announcements of new, moved and removed visuals, the overview's headings. They are in the visual's own language (a
 * Hindi visual is announced in Hindi even when the app around it is English). The rest of the words (panels, tooltips,
 * tables) live in labels-panel.ts, which loads with the chart code.
 */

import type { CanvasLanguage } from "./types";

export interface ShellLabels {
  preparing: string;
  failed: string;
  dismiss: string;
  retry: string;
  position: (i: number, n: number) => string;
  moved: (title: string, i: number, n: number) => string;
  removed: (title: string) => string;
  added: (title: string, summary: string) => string;
  overview: string;
  overviewBuilding: string;
  panel: string;
  visuals: string;
}

const EN: ShellLabels = {
  preparing: "Preparing a visual…",
  failed: "A visual couldn't be built this time. The answer is unaffected.",
  dismiss: "Dismiss",
  retry: "Try again",
  position: (i, n) => `${i} of ${n}`,
  moved: (title, i, n) => `${title} moved to position ${i} of ${n}`,
  removed: (title) => `Removed from the canvas: ${title}`,
  added: (title, summary) => `New on the canvas: ${title}. ${summary}`,
  overview: "Overview",
  overviewBuilding: "Building the overview…",
  panel: "Visual",
  visuals: "Visuals",
};

const HI: ShellLabels = {
  preparing: "विज़ुअल तैयार हो रहा है…",
  failed: "इस बार विज़ुअल नहीं बन सका। जवाब पर इसका असर नहीं पड़ा।",
  dismiss: "बंद करें",
  retry: "फिर कोशिश करें",
  position: (i, n) => `${n} में से ${i}`,
  moved: (title, i, n) => `${title} को ${n} में से ${i} स्थान पर ले जाया गया`,
  removed: (title) => `कैनवास से हटाया गया: ${title}`,
  added: (title, summary) => `कैनवास पर नया: ${title}। ${summary}`,
  overview: "सार",
  overviewBuilding: "सार तैयार हो रहा है…",
  panel: "विज़ुअल",
  visuals: "विज़ुअल",
};

export const shellLabelsFor = (language: CanvasLanguage | string | null | undefined): ShellLabels => (language === "hi" ? HI : EN);
