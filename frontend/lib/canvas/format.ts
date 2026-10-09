/**
 * Number, unit and date formatting for the canvas. The backend sends numbers in the unit's scale (4210 with
 * `scale: "crore"` is ₹4,210 crore); this file only writes them down. It never computes a figure.
 *
 *   - Indian grouping (12,34,567) whenever the source is INR or uses lakh/crore, western grouping otherwise.
 *   - The scale word is spelled out after the number ("₹4,210 crore", "₹4,210 करोड़"), never folded into the number.
 *   - Digits stay Latin in Hindi, as the documents print them; only words and dates are localised.
 *   - At most two decimals, trailing zeros dropped, so what is shown stays close to the cell it came from.
 *   - Negative numbers use a true minus sign (−). Screen reader text spells it out.
 */

import type { CanvasLanguage, DeltaKind, Unit, UnitScale } from "./types";

export const MINUS = "−";
export const EN_DASH_NONE = "—";

export type Grouping = "indian" | "western";

const SCALE_WORDS: Record<CanvasLanguage, Record<UnitScale, string>> = {
  en: { crore: "crore", lakh: "lakh", million: "million", billion: "billion", thousand: "thousand" },
  hi: { crore: "करोड़", lakh: "लाख", million: "मिलियन", billion: "बिलियन", thousand: "हज़ार" },
};

const CURRENCY_FALLBACK: Record<string, string> = { INR: "₹", USD: "$", EUR: "€", GBP: "£", JPY: "¥", CNY: "¥" };

const symbolCache = new Map<string, string>();

/** "₹" for INR, "$" for USD, the code itself ("CHF") when the platform has no narrow symbol. */
export function currencySymbol(code: string | null): string {
  if (!code) return "";
  const hit = symbolCache.get(code);
  if (hit !== undefined) return hit;
  let symbol = CURRENCY_FALLBACK[code] ?? code;
  if (!CURRENCY_FALLBACK[code]) {
    try {
      const part = new Intl.NumberFormat("en", { style: "currency", currency: code, currencyDisplay: "narrowSymbol" })
        .formatToParts(0)
        .find((p) => p.type === "currency");
      if (part?.value) symbol = part.value;
    } catch {
      // not an ISO code: keep it as written
    }
  }
  symbolCache.set(code, symbol);
  return symbol;
}

export function groupingFor(unit: Unit | null | undefined): Grouping {
  if (!unit) return "western";
  return unit.currency === "INR" || unit.scale === "crore" || unit.scale === "lakh" ? "indian" : "western";
}

const formatters = new Map<string, Intl.NumberFormat>();

function formatterFor(grouping: Grouping, maxFraction: number): Intl.NumberFormat {
  const key = `${grouping}:${maxFraction}`;
  let f = formatters.get(key);
  if (!f) {
    f = new Intl.NumberFormat(grouping === "indian" ? "en-IN" : "en-US", {
      maximumFractionDigits: maxFraction,
      minimumFractionDigits: 0,
      useGrouping: true,
    });
    formatters.set(key, f);
  }
  return f;
}

/** The magnitude of `value` written with grouping and at most `maxFraction` decimals (no sign). */
export function formatMagnitude(value: number, grouping: Grouping = "western", maxFraction = 2): string {
  if (!Number.isFinite(value)) return EN_DASH_NONE;
  return formatterFor(grouping, maxFraction).format(Math.abs(value));
}

/** True when `value` would print as zero at this precision (so "-0" never shows). */
function isZeroAt(value: number, maxFraction: number): boolean {
  return Math.abs(value) < 0.5 * 10 ** -maxFraction;
}

export interface FormatOptions {
  /** Write the unit's symbol and scale word (default). False: the number alone (axis ticks). */
  withUnit?: boolean;
  /** "+" before positive numbers. */
  signed?: boolean;
  /** Screen reader wording: the sign as a word, the missing value as text. */
  spoken?: boolean;
  maxFraction?: number;
}

function signOf(value: number, maxFraction: number, signed: boolean, spoken: boolean, language: CanvasLanguage): string {
  if (isZeroAt(value, maxFraction)) return "";
  const word = language === "hi" ? "माइनस " : "minus ";
  if (value < 0) return spoken ? word : MINUS;
  return signed ? (spoken ? (language === "hi" ? "प्लस " : "plus ") : "+") : "";
}

export interface ValueParts {
  /** "−", "+" or "" (spoken: "minus "). It leads the whole amount: "−₹1,240 crore". */
  sign: string;
  /** Currency symbol, "" otherwise. */
  prefix: string;
  /** The grouped number. */
  number: string;
  /** Scale word, "%", "×" or a unit label, "" when there is none. */
  suffix: string;
  /** The suffix touches the number ("18.4%", "1.35×") instead of following a space ("4,210 crore"). */
  tight: boolean;
}

/** A value split for display (a big number with a small unit beside it), or null for a missing one. */
export function formatParts(
  value: number | null | undefined,
  unit: Unit | null | undefined,
  language: CanvasLanguage = "en",
  opts: FormatOptions = {},
): ValueParts | null {
  const { withUnit = true, signed = false, spoken = false, maxFraction = 2 } = opts;
  if (value === null || value === undefined || !Number.isFinite(value)) return null;
  const sign = signOf(value, maxFraction, signed, spoken, language);
  const number = formatMagnitude(value, groupingFor(unit), maxFraction);
  const parts: ValueParts = { sign, prefix: "", number, suffix: "", tight: false };
  if (!withUnit || !unit) return parts;
  switch (unit.kind) {
    case "currency": {
      const symbol = currencySymbol(unit.currency);
      parts.prefix = symbol && /^\p{L}+$/u.test(symbol) && symbol.length > 1 ? `${symbol} ` : symbol;
      parts.suffix = unit.scale ? SCALE_WORDS[language][unit.scale] : "";
      break;
    }
    case "percent":
      parts.suffix = "%";
      parts.tight = true;
      break;
    case "ratio":
      parts.suffix = "×";
      parts.tight = true;
      break;
    default: {
      const scale = unit.scale ? SCALE_WORDS[language][unit.scale] : "";
      const label = unit.label && unit.label.toLowerCase() !== "count" ? unit.label : "";
      parts.suffix = [scale, label].filter(Boolean).join(" ");
    }
  }
  return parts;
}

/** Value with its unit: "₹4,210 crore", "18.4%", "1.35×", "90 days", "12,34,567". `null` is "—". */
export function formatValue(
  value: number | null | undefined,
  unit: Unit | null | undefined,
  language: CanvasLanguage = "en",
  opts: FormatOptions = {},
): string {
  const p = formatParts(value, unit, language, opts);
  if (!p) return opts.spoken ? (language === "hi" ? "कोई मान नहीं" : "no value") : EN_DASH_NONE;
  return `${p.sign}${p.prefix}${p.number}${p.suffix ? (p.tight ? "" : " ") + p.suffix : ""}`;
}

/** What an axis shows beside its ticks: "₹ crore", "%", "days", "" when there is nothing to say. */
export function unitCaption(unit: Unit | null | undefined, language: CanvasLanguage = "en"): string {
  if (!unit) return "";
  switch (unit.kind) {
    case "currency": {
      const symbol = currencySymbol(unit.currency);
      const scale = unit.scale ? SCALE_WORDS[language][unit.scale] : "";
      const built = [symbol, scale].filter(Boolean).join(" ");
      return built || unit.label;
    }
    case "percent":
      return "%";
    case "ratio":
      return "×";
    default: {
      const scale = unit.scale ? SCALE_WORDS[language][unit.scale] : "";
      const label = unit.label && unit.label.toLowerCase() !== "count" ? unit.label : "";
      return [scale, label].filter(Boolean).join(" ");
    }
  }
}

/** Two series share an axis only when this is equal. */
export function unitSignature(unit: Unit | null | undefined): string {
  if (!unit) return "none";
  return [unit.kind, unit.currency ?? "", unit.scale ?? "", unit.kind === "none" || unit.kind === "count" || unit.kind === "duration" ? unit.label : ""].join("|");
}

/** An axis tick: the number, plus "%" for percentages (the unit caption says the rest). */
export function formatTick(value: number, unit: Unit | null | undefined): string {
  const mag = formatMagnitude(value, groupingFor(unit), 2);
  const sign = isZeroAt(value, 2) ? "" : value < 0 ? MINUS : "";
  return `${sign}${mag}${unit?.kind === "percent" ? "%" : ""}`;
}

/** A figure in a table body: the number alone (its column header carries the unit), with "%" and "×" kept. */
export function formatCell(value: number | null | undefined, unit: Unit | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EN_DASH_NONE;
  return unit?.kind === "ratio" ? formatValue(value, unit) : formatTick(value, unit);
}

const DELTA_PP: Record<CanvasLanguage, string> = { en: "pp", hi: "प्रतिशत अंक" };

/** A change: "+12.3%", "−1.2 pp", "+₹520 crore". `kind` says how to read the number. */
export function formatDelta(
  value: number,
  kind: DeltaKind,
  unit: Unit | null | undefined,
  language: CanvasLanguage = "en",
  opts: { spoken?: boolean } = {},
): string {
  const spoken = opts.spoken === true;
  if (kind === "pct") return formatValue(value, { kind: "percent", currency: null, scale: null, label: "%" }, language, { signed: true, spoken });
  if (kind === "pp") {
    const num = formatValue(value, null, language, { signed: true, spoken });
    return `${num} ${DELTA_PP[language]}`;
  }
  return formatValue(value, unit, language, { signed: true, spoken });
}

/**
 * The unit a calculation's value is written in: its own, or (when the backend left it out) what its operation implies:
 * growth, CAGR and share are percentages, a ratio is a ratio, anything else follows the visual.
 */
export function calcUnit(op: string, unit: Unit | null | undefined, fallback: Unit | null | undefined): Unit | null {
  if (unit) return unit;
  if (op === "growth" || op === "cagr" || op === "share") return { kind: "percent", currency: null, scale: null, label: "%" };
  if (op === "ratio") return { kind: "ratio", currency: null, scale: null, label: "x" };
  return fallback ?? null;
}

export type Direction = "up" | "down" | "flat";

/** Which way a change points (for the arrow beside it). */
export function directionOf(value: number): Direction {
  if (isZeroAt(value, 2)) return "flat";
  return value > 0 ? "up" : "down";
}

// ------------------------------------------------------------------ dates

const ISO_DATE = /^(\d{4})-(\d{2})-(\d{2})$/;

export const isIsoDate = (s: string): boolean => {
  const m = ISO_DATE.exec(s);
  if (!m) return false;
  const d = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3]));
  return d.getUTCFullYear() === +m[1] && d.getUTCMonth() === +m[2] - 1 && d.getUTCDate() === +m[3];
};

/** Milliseconds since the epoch for "YYYY-MM-DD", or null. */
export const isoToTime = (s: string): number | null => {
  if (!isIsoDate(s)) return null;
  const m = ISO_DATE.exec(s)!;
  return Date.UTC(+m[1], +m[2] - 1, +m[3]);
};

const dateFormatters = new Map<string, Intl.DateTimeFormat>();

/** "15 Mar 2025" / "15 मार्च 2025" for an ISO date; anything else is returned as written. UTC, so no day shifts. */
export function formatDate(date: string, language: CanvasLanguage = "en", style: "short" | "long" = "short"): string {
  const t = isoToTime(date);
  if (t === null) return date;
  const key = `${language}:${style}`;
  let f = dateFormatters.get(key);
  if (!f) {
    f = new Intl.DateTimeFormat(language === "hi" ? "hi-IN-u-nu-latn" : "en-IN", {
      day: "numeric",
      month: style === "long" ? "long" : "short",
      year: "numeric",
      timeZone: "UTC",
    });
    dateFormatters.set(key, f);
  }
  return f.format(new Date(t));
}
