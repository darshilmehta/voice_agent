/**
 * Words the panels show around the numbers, in the visual's own language (a Hindi visual reads Hindi even when the app
 * around it is English). Numbers keep Latin digits, as the documents print them. Loaded with the chart code; the words
 * the shell needs earlier are in labels.ts.
 */

import { shellLabelsFor, type ShellLabels } from "./labels";
import type { CanvasLanguage, VisualKind } from "./types";

export interface Labels extends ShellLabels {
  calculated: string;
  calculatedHint: string;
  viewTable: string;
  viewChart: string;
  source: string;
  page: string;
  cell: string;
  noValue: string;
  notReported: string;
  pin: string;
  unpin: string;
  pinned: string;
  remove: string;
  moveUp: string;
  moveDown: string;
  drag: string;
  highlighted: string;
  calculations: string;
  formula: string;
  inputs: string;
  sources: string;
  total: string;
  increase: string;
  decrease: string;
  series: string;
  clickForSource: string;
  showAll: (n: number) => string;
  showFewer: string;
  showing: (shown: number, total: number) => string;
  updating: string;
  unsupported: (kind: string) => string;
  donutTooMany: string;
  chartKind: Record<VisualKind, string>;
  chartKeys: string;
  empty: string;
}

type PanelLabels = Omit<Labels, keyof ShellLabels>;

const EN: PanelLabels = {
  calculated: "calculated",
  calculatedHint: "Calculated from figures in the document, not read from a cell",
  viewTable: "View as table",
  viewChart: "View as chart",
  source: "Source",
  page: "Page",
  cell: "Cell",
  noValue: "no value",
  notReported: "Not reported",
  pin: "Pin to canvas",
  unpin: "Unpin",
  pinned: "Pinned",
  remove: "Remove from canvas",
  moveUp: "Move up",
  moveDown: "Move down",
  drag: "Drag to reorder",
  highlighted: "Highlighted",
  calculations: "How these were calculated",
  formula: "Formula",
  inputs: "Inputs",
  sources: "Sources",
  total: "Total",
  increase: "Increase",
  decrease: "Decrease",
  series: "Series",
  clickForSource: "Click for the source",
  showAll: (n) => `Show all ${n}`,
  showFewer: "Show fewer",
  showing: (shown, total) => `Showing ${shown} of ${total}`,
  updating: "Updating…",
  unsupported: (kind) => `This version can't draw “${kind}” yet, so it is shown as a table.`,
  donutTooMany: "Too many parts for a ring, so they are shown as bars.",
  chartKind: {
    kpi: "Key figures",
    line: "Line chart",
    bar: "Bar chart",
    grouped_bar: "Grouped bar chart",
    stacked_bar: "Stacked bar chart",
    waterfall: "Waterfall chart",
    donut: "Donut chart",
    table: "Table",
    comparison: "Comparison",
    timeline: "Timeline",
  },
  chartKeys: "Arrow keys move between data points. Enter shows the source.",
  empty: "No data to show.",
};

const HI: PanelLabels = {
  calculated: "गणना की गई",
  calculatedHint: "दस्तावेज़ के आँकड़ों से निकाला गया, किसी सेल से सीधे पढ़ा नहीं गया",
  viewTable: "तालिका के रूप में देखें",
  viewChart: "चार्ट के रूप में देखें",
  source: "स्रोत",
  page: "पृष्ठ",
  cell: "सेल",
  noValue: "कोई मान नहीं",
  notReported: "उपलब्ध नहीं",
  pin: "कैनवास पर पिन करें",
  unpin: "पिन हटाएँ",
  pinned: "पिन किया हुआ",
  remove: "कैनवास से हटाएँ",
  moveUp: "ऊपर ले जाएँ",
  moveDown: "नीचे ले जाएँ",
  drag: "क्रम बदलने के लिए खींचें",
  highlighted: "ख़ास",
  calculations: "यह कैसे निकाला गया",
  formula: "सूत्र",
  inputs: "इनपुट",
  sources: "स्रोत",
  total: "कुल",
  increase: "बढ़त",
  decrease: "गिरावट",
  series: "श्रृंखला",
  clickForSource: "स्रोत के लिए क्लिक करें",
  showAll: (n) => `सभी ${n} दिखाएँ`,
  showFewer: "कम दिखाएँ",
  showing: (shown, total) => `${total} में से ${shown} दिख रहे हैं`,
  updating: "अपडेट हो रहा है…",
  unsupported: (kind) => `यह संस्करण “${kind}” अभी नहीं बना सकता, इसलिए तालिका दिखाई गई है।`,
  donutTooMany: "रिंग के लिए हिस्से बहुत ज़्यादा हैं, इसलिए बार में दिखाए गए हैं।",
  chartKind: {
    kpi: "मुख्य आँकड़े",
    line: "लाइन चार्ट",
    bar: "बार चार्ट",
    grouped_bar: "समूहित बार चार्ट",
    stacked_bar: "स्टैक्ड बार चार्ट",
    waterfall: "वॉटरफ़ॉल चार्ट",
    donut: "डोनट चार्ट",
    table: "तालिका",
    comparison: "तुलना",
    timeline: "समयरेखा",
  },
  chartKeys: "डेटा बिंदुओं के बीच जाने के लिए तीर कुंजियाँ दबाएँ। स्रोत देखने के लिए Enter दबाएँ।",
  empty: "दिखाने के लिए कोई डेटा नहीं।",
};

const cache = new Map<string, Labels>();

/** Every word a panel needs, in the visual's language. */
export function labelsFor(language: CanvasLanguage | string | null | undefined): Labels {
  const key = language === "hi" ? "hi" : "en";
  let hit = cache.get(key);
  if (!hit) {
    hit = { ...shellLabelsFor(key), ...(key === "hi" ? HI : EN) };
    cache.set(key, hit);
  }
  return hit;
}
