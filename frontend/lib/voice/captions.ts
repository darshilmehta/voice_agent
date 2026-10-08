/**
 * Live captions: the words of what the agent says, with the timing to highlight each as it is spoken.
 *
 * The protocol announces each synthesized clause as `audio_chunk {text, duration_ms}` before its audio. Inside a chunk
 * a word is spoken in proportion to its length (letters, plus a little pause after punctuation), a good stand-in for
 * the per-token durations the TTS has (docs §3.8); across chunks the real playback clock decides, so captions can't
 * run ahead of the audio or drift from it: they follow what the AudioContext is actually playing.
 */

import { splitCitations } from "../citations";

export interface CaptionWord {
  text: string;
  /** Index of the announced chunk the word belongs to, or -1 for text that has no audio yet. */
  chunk: number;
  /** Where in its chunk (0…1) the word starts being spoken. */
  at: number;
}

/** Text as it is spoken: `[S1]` markers removed, spacing before punctuation tidied. */
export function plainSpeech(text: string): string {
  return splitCitations(text)
    .map((p) => (p.kind === "text" ? p.text : ""))
    .join("")
    .replace(/\s+([.,;:!?।])/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

const PAUSE = /[.,;:!?।…)]$/;

/** Relative time a word takes to say. */
function weight(word: string): number {
  const letters = word.replace(/[^\p{L}\p{N}\p{M}]/gu, "").length;
  const pause = /[.!?।…]$/.test(word) ? 3 : PAUSE.test(word) ? 1.5 : 0;
  return Math.max(1, letters) + pause;
}

/** Words of a chunk with the fraction of the chunk at which each starts. */
export function chunkWords(text: string, chunk: number): CaptionWord[] {
  const words = plainSpeech(text).split(" ").filter(Boolean);
  const weights = words.map(weight);
  const total = weights.reduce((a, b) => a + b, 0) || 1;
  let acc = 0;
  return words.map((w, i) => {
    const at = acc / total;
    acc += weights[i];
    return { text: w, chunk, at };
  });
}

export interface ChunkText {
  index: number;
  text: string;
}

/**
 * All the words of a turn: the announced chunks (with timing), then any generated text that hasn't been announced
 * yet (shown dimmed; it may be ahead of the audio, or be the whole answer if no audio ever comes).
 */
export function turnWords(chunks: ChunkText[], deltaText: string): CaptionWord[] {
  const words: CaptionWord[] = [];
  for (const c of chunks) words.push(...chunkWords(c.text, c.index));
  const spoken = plainSpeech(chunks.map((c) => c.text).join(" "));
  const generated = plainSpeech(deltaText);
  if (generated.length > spoken.length && generated.startsWith(spoken)) {
    for (const w of generated.slice(spoken.length).split(" ").filter(Boolean)) words.push({ text: w, chunk: -1, at: 0 });
  } else if (chunks.length === 0 && generated) {
    for (const w of generated.split(" ")) words.push({ text: w, chunk: -1, at: 0 });
  }
  return words;
}

/**
 * How many leading words have been (at least begun to be) spoken, given how far into each chunk the voice is.
 * `fraction(chunk)` is 0 for a chunk whose audio hasn't started.
 */
export function spokenCount(words: CaptionWord[], fraction: (chunk: number) => number): number {
  // The last word whose time has come; a later chunk that has started means every earlier word is behind us.
  let n = 0;
  words.forEach((w, i) => {
    if (w.chunk < 0) return;
    const f = fraction(w.chunk);
    if (f > 0 && f >= w.at - 1e-6) n = i + 1;
  });
  return n;
}
