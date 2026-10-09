/**
 * Live captions: the words of what the agent says, with the timing to highlight each as it is spoken.
 *
 * The protocol announces each synthesized clause as `audio_chunk {text, duration_ms}` before its audio. Inside a chunk
 * a word is spoken in proportion to its length (letters, plus a little pause after punctuation), a good stand-in for
 * the per-token durations the TTS has (docs §3.8); across chunks the real playback clock decides, so captions can't
 * run ahead of the audio or drift from it: they follow what the AudioContext is actually playing.
 *
 * A turn that searches the web speaks a filler first ("Let me look that up.", `audio_chunk {filler: true}`, docs §3.7).
 * It is not part of the answer (the saved text and `heard_text` leave it out), so it never takes part in comparing
 * what was spoken with what was generated; its words are marked `aside` and shown as a light line before the answer.
 */

import { splitCitations } from "../citations";

export interface CaptionWord {
  text: string;
  /** Index of the announced chunk the word belongs to, or -1 for text that has no audio yet. */
  chunk: number;
  /** Where in its chunk (0…1) the word starts being spoken. */
  at: number;
  /** A word of the filler, not of the answer. */
  aside?: boolean;
}

/** Text as it is spoken: `[S1]` and `[W1]` markers removed, spacing before punctuation tidied. */
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
export function chunkWords(text: string, chunk: number, aside = false): CaptionWord[] {
  const words = plainSpeech(text).split(" ").filter(Boolean);
  const weights = words.map(weight);
  const total = weights.reduce((a, b) => a + b, 0) || 1;
  let acc = 0;
  return words.map((w, i) => {
    const at = acc / total;
    acc += weights[i];
    return aside ? { text: w, chunk, at, aside } : { text: w, chunk, at };
  });
}

export interface ChunkText {
  index: number;
  text: string;
  /** The filler spoken while the web is searched: not part of the answer. */
  filler?: boolean;
}

/**
 * All the words of a turn: the announced chunks (with timing), then any generated text that hasn't been announced
 * yet (shown dimmed; it may be ahead of the audio, or be the whole answer if no audio ever comes). The filler's words
 * come with the chunks (they are spoken, and light up as they are) but are left out of the comparison of what was
 * spoken with what was generated: the generated text never contains them.
 */
export function turnWords(chunks: ChunkText[], deltaText: string): CaptionWord[] {
  const words: CaptionWord[] = [];
  for (const c of chunks) words.push(...chunkWords(c.text, c.index, c.filler === true));
  const answer = chunks.filter((c) => !c.filler);
  const spoken = plainSpeech(answer.map((c) => c.text).join(" "));
  const generated = plainSpeech(deltaText);
  if (generated.length > spoken.length && generated.startsWith(spoken)) {
    for (const w of generated.slice(spoken.length).split(" ").filter(Boolean)) words.push({ text: w, chunk: -1, at: 0 });
  } else if (answer.length === 0 && generated) {
    for (const w of generated.split(" ")) words.push({ text: w, chunk: -1, at: 0 });
  }
  return words;
}

/** What the captions need to know about the agent's turn. */
export interface TurnCaptionInput {
  chunks: ChunkText[];
  deltaText: string;
  /** The saved answer, with what was actually heard when it was cut off. */
  message: { text: string; heard_text: string | null } | null;
}

/**
 * The words the agent's captions show for a turn, and whether the server's `heard_text` is all there is to show
 * (cut off, and no answer audio was ever announced). The filler's words lead, as an aside; text that has no timing
 * of its own (an answer spoken without announced chunks) gets a chunk index no filler uses.
 */
export function agentCaptionWords(t: TurnCaptionInput): { words: CaptionWord[]; trustServer: boolean } {
  const answer = t.chunks.filter((c) => !c.filler);
  const heard = t.message?.heard_text ?? null;
  const trustServer = !!t.message && heard !== null && answer.length === 0;
  const untimed = t.chunks.reduce((n, c) => Math.max(n, c.index + 1), 0);
  const filler = () => t.chunks.filter((c) => c.filler).flatMap((c) => chunkWords(c.text, c.index, true));
  if (trustServer) return { words: [...filler(), ...chunkWords(heard ?? "", untimed)], trustServer };
  // Spoken without caption timing: the saved text is all there is.
  if (t.message && answer.length === 0) return { words: [...filler(), ...chunkWords(t.message.text, untimed)], trustServer };
  return { words: turnWords(t.chunks, t.deltaText), trustServer };
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
