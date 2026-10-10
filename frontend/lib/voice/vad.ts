/**
 * Browser voice-activity detection for barge-in (docs/DESIGN.md §3.3, §9.5): Silero v5 through @ricky0123/vad-web's
 * MicVAD, running on the same microphone stream and AudioContext as the uplink. It tells the page the moment the user
 * starts speaking, so the agent can be ducked locally (< 200 ms) before the server has heard a word.
 *
 * vad-web and onnxruntime-web are served from this app (`public/vad/`, copied from node_modules by
 * scripts/copy-vad-assets.mjs) and loaded here as two classic scripts, on first use only. Nothing comes from a CDN,
 * and the ~14 MB WASM binary stays out of the app bundle (bundlers handle onnxruntime's dynamic WASM import badly).
 */

import type { MicVAD } from "@ricky0123/vad-web";

import type { PublicVoiceInput } from "../api";
import {
  DEFAULT_VOICE_INPUT,
  frameDbfs,
  LEVEL_PERCENTILE,
  nearFieldMin,
  NoiseFloor,
  opensSpeech,
  percentile,
  UserLevel,
  vadTuning,
} from "./noise";

const ASSET_BASE = "/vad/";

// What the two scripts put on window: `ort` (onnxruntime-web, WASM only) and `vad` (vad-web's UMD bundle).
declare global {
  interface Window {
    ort?: { env: { wasm: { numThreads?: number }; logLevel?: string } };
    vad?: { MicVAD: typeof MicVAD };
  }
}

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const el = document.createElement("script");
    el.src = src;
    el.async = false;
    el.onload = () => resolve();
    el.onerror = () => {
      el.remove();
      reject(new Error(`Couldn't load ${src}`));
    };
    document.head.appendChild(el);
  });
}

let scripts: Promise<void> | null = null;

/** Loads the VAD scripts once per page. A failed load can be retried by the next call. */
export function loadVadScripts(): Promise<void> {
  if (window.vad) return Promise.resolve();
  scripts ??= (async () => {
    await loadScript(`${ASSET_BASE}ort.wasm.min.js`); // window.ort, which vad's bundle expects to find
    await loadScript(`${ASSET_BASE}bundle.min.js`); // window.vad
  })().catch((err) => {
    scripts = null;
    throw err;
  });
  return scripts;
}

export interface BargeInVadEvents {
  /** Speech began (the very first speech frame): duck the agent. */
  onSpeechStart: () => void;
  /** Speech lasted long enough to be real (`minSpeechMs`). */
  onSpeechRealStart: () => void;
  /** It was too short to be speech (a cough, a click, "mm"): restore the volume. */
  onMisfire: () => void;
  onSpeechEnd: () => void;
}

export interface BargeInVad {
  destroy: () => Promise<void>;
  /** The last speech segment was the user's (their message was saved): learn their level from it. */
  noteUserTurn: () => void;
}

/**
 * Frees a VAD whose `start()` failed. MicVAD.destroy() assumes the audio nodes exist and throws before it releases the
 * model when they don't (the failure came while they were being made), so the model is released directly then.
 */
async function releaseUnstarted(vad: MicVAD): Promise<void> {
  try {
    await vad.destroy();
    return;
  } catch {
    // no audio nodes to tear down
  }
  try {
    await (vad as unknown as { model: { release: () => Promise<void> } }).model.release();
  } catch {
    // already released
  }
}

/** How often (frames of 32 ms) the VAD's thresholds follow the noise floor. */
const RETUNE_FRAMES = 16;

/**
 * Starts the VAD on an existing stream and context. Tuning follows the smoke tests (§9.5): thresholds 0.5 / 0.35,
 * ~600 ms of trailing silence ends a segment, anything under ~250 ms is a misfire.
 *
 * In a noisy room (docs/DESIGN.md §3.10, `input.adaptive_gating`) the thresholds follow the noise floor (lib/voice/
 * noise.ts: the same gate as the server's), and speech only ducks the agent (`onSpeechStart`) once one of its frames
 * is `start_snr_db` above the floor and the segment is loud enough to be the user's (`nearFieldMin`: their level,
 * learnt from their turns): a fan, traffic, the café's babble or the TV don't, the user close to the mic does.
 */
export async function startBargeInVad(
  ctx: AudioContext,
  stream: MediaStream,
  events: BargeInVadEvents,
  input: PublicVoiceInput = DEFAULT_VOICE_INPUT,
): Promise<BargeInVad> {
  await loadVadScripts();
  const lib = window.vad;
  if (!lib) throw new Error("The VAD script didn't register");
  const base = new URL(ASSET_BASE, window.location.origin).href;

  // The gate: the VAD's own segment (onSpeechStart … onSpeechEnd / onVADMisfire) is reported only once a frame of it
  // passes `opensSpeech`; a segment that never does was the room, and ends silently.
  const floor = new NoiseFloor(input.floor_window_ms, input.floor_percentile);
  let inSegment = false;
  let confirmed = false;
  let frames = 0;
  let tuned = vadTuning(floor.dbfs, input);
  let micVad: MicVAD | null = null;
  let last = { p: 0, dbfs: -100 }; // the frame vad-web reports before its SpeechStart
  const user = new UserLevel();
  let segment: number[] = []; // levels of the segment's speech frames
  let lastSegment: number[] = [];
  const confirm = (p: number, dbfs: number) => {
    if (!inSegment) return;
    if (p >= input.threshold) segment.push(dbfs);
    if (confirmed || !opensSpeech(p, dbfs, floor.dbfs, input)) return;
    if (input.adaptive_gating && percentile(segment, LEVEL_PERCENTILE) < nearFieldMin(input, user.dbfs)) return;
    confirmed = true;
    events.onSpeechStart();
  };
  const onFrameProcessed = (probs: { isSpeech: number }, frame: Float32Array) => {
    const dbfs = frameDbfs(frame);
    floor.push(dbfs);
    last = { p: probs.isSpeech, dbfs };
    if (inSegment && segment.length > 0) confirm(probs.isSpeech, dbfs); // (the onset frame: from onSpeechStart)
    if (input.adaptive_gating && ++frames % RETUNE_FRAMES === 0) {
      const next = vadTuning(floor.dbfs, input);
      if (Math.abs(next.positiveSpeechThreshold - tuned.positiveSpeechThreshold) >= 0.02 || next.minSpeechMs !== tuned.minSpeechMs) {
        tuned = next;
        micVad?.setOptions(next);
      }
    }
  };
  const segmentOver = (report: () => void) => {
    const was = confirmed;
    inSegment = confirmed = false;
    if (segment.length > 0) lastSegment = segment;
    segment = [];
    if (was) report();
  };

  const vad = await lib.MicVAD.new({
    model: "v5",
    audioContext: ctx,
    baseAssetPath: base,
    onnxWASMBasePath: base,
    // The page owns the microphone: the VAD listens to its stream and never opens or stops one itself.
    getStream: async () => stream,
    pauseStream: async () => undefined,
    resumeStream: async () => stream,
    startOnLoad: false,
    positiveSpeechThreshold: tuned.positiveSpeechThreshold,
    negativeSpeechThreshold: tuned.negativeSpeechThreshold,
    redemptionMs: 600,
    minSpeechMs: tuned.minSpeechMs,
    ortConfig: (ort: NonNullable<Window["ort"]>) => {
      ort.env.logLevel = "error";
      ort.env.wasm.numThreads = 1; // no cross-origin isolation here, so no threads
    },
    onFrameProcessed,
    onSpeechStart: () => {
      inSegment = true;
      confirmed = false;
      segment = [];
      confirm(last.p, last.dbfs); // the onset frame itself (always, in a quiet room), else a later one of the segment
    },
    onSpeechRealStart: () => {
      if (confirmed) events.onSpeechRealStart();
    },
    onVADMisfire: () => segmentOver(events.onMisfire),
    onSpeechEnd: () => segmentOver(events.onSpeechEnd),
  } as Parameters<typeof MicVAD.new>[0]);
  micVad = vad;
  try {
    await vad.start();
  } catch (err) {
    // The model and its ONNX session are loaded by now: release them, or they stay in memory for good.
    await releaseUnstarted(vad);
    throw err;
  }
  return {
    destroy: () => vad.destroy().catch(() => undefined),
    noteUserTurn: () => {
      const levels = segment.length > 0 ? segment : lastSegment;
      if (levels.length > 0) user.update(percentile(levels, LEVEL_PERCENTILE));
      lastSegment = [];
    },
  };
}
