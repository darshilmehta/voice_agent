"use client";

/**
 * Text and element measuring for the charts. Text is measured with a canvas in the page's own font (so Devanagari and
 * Latin both fit what is drawn); before that is possible, or when the browser can't, a width estimate is used
 * (lib/canvas/scales.ts). Only the DOM-bound parts are here; the rest is pure.
 */

import { useEffect, useLayoutEffect, useMemo, useRef, useState, type RefObject } from "react";

import { estimateWidth, type Measure } from "@/lib/canvas/scales";

let context: CanvasRenderingContext2D | null | undefined;

function ctx(): CanvasRenderingContext2D | null {
  if (context !== undefined) return context;
  try {
    context = document.createElement("canvas").getContext("2d");
  } catch {
    context = null;
  }
  return context;
}

const cache = new Map<string, number>();

/** A text measurer for one font size and weight in the page's font family. */
export function makeMeasure(size: number, weight = 400): Measure {
  const family = typeof document !== "undefined" ? getComputedStyle(document.body).fontFamily : "sans-serif";
  const font = `${weight} ${size}px ${family}`;
  return (text) => {
    const c = ctx();
    if (!c) return estimateWidth(text, size);
    const key = `${font}|${text}`;
    const hit = cache.get(key);
    if (hit !== undefined) return hit;
    c.font = font;
    const w = c.measureText(text).width;
    if (cache.size > 4000) cache.clear();
    cache.set(key, w);
    return w;
  };
}

/** A measurer that follows the document's fonts: made again when they finish loading. */
export function useMeasure(size: number, weight = 400): Measure {
  const [epoch, setEpoch] = useState(0);
  useEffect(() => {
    const fonts = (document as Document & { fonts?: FontFaceSet }).fonts;
    if (!fonts) return;
    let alive = true;
    void fonts.ready.then(() => {
      if (alive) {
        cache.clear();
        setEpoch((n) => n + 1);
      }
    });
    return () => {
      alive = false;
    };
  }, []);
  // `epoch` only forces a new measurer after fonts load
  return useMemo(() => makeMeasure(size, weight), [size, weight, epoch]);
}

/** The width of an element, kept current as it resizes (0 until the first measurement). */
export function useElementWidth(ref: RefObject<HTMLElement | null>): number {
  const [width, setWidth] = useState(0);
  const last = useRef(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const read = () => {
      const w = Math.floor(el.getBoundingClientRect().width);
      if (w !== last.current) {
        last.current = w;
        setWidth(w);
      }
    };
    read();
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return width;
}
