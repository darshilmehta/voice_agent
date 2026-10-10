/**
 * Microphone capture for the voice session: getUserMedia (echo cancellation, noise suppression, auto gain), an
 * AudioWorklet that downsamples to 16 kHz and cuts PCM16 frames, and an AnalyserNode that feeds the presence field
 * with the user's voice (after the browser's echo cancellation, so the agent's own voice doesn't drive the user's waves).
 */

import { createDenoiser, type Denoiser } from "./denoise";
import { INPUT_SAMPLE_RATE, UPLINK_FRAME_SAMPLES } from "./protocol";

/**
 * The worklet's source. It lives in a string, not a file, so the app doesn't depend on a bundler or a public path for
 * it; it is loaded from a Blob URL.
 *
 * Downsampling to 16 kHz is a windowed-sinc low-pass fused with the decimation: each output sample is a 32-tap FIR
 * (Kaiser window, beta 6) of the input around its instant, with the cutoff at 7 kHz and 256 fractional phases so any
 * input rate (48, 44.1, 32 kHz …) lands exactly on the 16 kHz grid. Anything above 8 kHz would fold back into the
 * speech band when decimated; this keeps it out (-60 dB or better from 10 kHz up, against -6 to -10 dB for a plain
 * average). Output timing is the same fixed ratio, so frames still leave every 32 ms of input; the only cost is 16
 * input samples (a third of a millisecond at 48 kHz) of look-ahead. At 16 kHz the input passes straight through.
 */
const WORKLET_SOURCE = `
class PcmCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = options.processorOptions || {};
    const target = o.targetRate || 16000;
    this.ratio = sampleRate / target;
    this.frame = o.frameSamples || 512;
    this.out = new Int16Array(this.frame);
    this.n = 0;
    this.direct = sampleRate === target;
    if (!this.direct) {
      const taps = 32, phases = 256, half = taps / 2, beta = 6;
      const cutoff = Math.min(7000, 0.45 * Math.min(sampleRate, target)) / sampleRate;
      const bessel = (x) => {
        let sum = 1, term = 1;
        for (let k = 1; k < 30; k++) { term *= (x / (2 * k)) * (x / (2 * k)); sum += term; }
        return sum;
      };
      const norm = bessel(beta);
      this.taps = taps;
      this.half = half;
      this.table = new Float32Array((phases + 1) * taps);
      for (let p = 0; p <= phases; p++) {
        let sum = 0;
        for (let j = 0; j < taps; j++) {
          const tau = j - (half - 1) - p / phases;
          const r = tau / half;
          const w = r >= 1 || r <= -1 ? 0 : bessel(beta * Math.sqrt(1 - r * r)) / norm;
          const a = 2 * cutoff * tau;
          const v = 2 * cutoff * (a === 0 ? 1 : Math.sin(Math.PI * a) / (Math.PI * a)) * w;
          this.table[p * taps + j] = v;
          sum += v;
        }
        for (let j = 0; j < taps; j++) this.table[p * taps + j] /= sum;
      }
      this.phases = phases;
      this.buf = new Float32Array(taps + 1024);
      this.len = half - 1;
      this.t = half - 1;
    }
  }
  emit(v) {
    v = v < -1 ? -1 : v > 1 ? 1 : v;
    this.out[this.n++] = v < 0 ? v * 32768 : v * 32767;
    if (this.n === this.frame) {
      const buffer = this.out.buffer;
      this.port.postMessage(buffer, [buffer]);
      this.out = new Int16Array(this.frame);
      this.n = 0;
    }
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    if (this.direct) {
      for (let i = 0; i < ch.length; i++) this.emit(ch[i]);
      return true;
    }
    if (this.len + ch.length > this.buf.length) {
      const bigger = new Float32Array(this.len + ch.length + 1024);
      bigger.set(this.buf.subarray(0, this.len));
      this.buf = bigger;
    }
    this.buf.set(ch, this.len);
    this.len += ch.length;
    const taps = this.taps, half = this.half, table = this.table, buf = this.buf;
    for (;;) {
      const i0 = Math.floor(this.t);
      if (i0 + half >= this.len) break;
      const row = Math.round((this.t - i0) * this.phases) * taps;
      const from = i0 - (half - 1);
      let acc = 0;
      for (let j = 0; j < taps; j++) acc += buf[from + j] * table[row + j];
      this.emit(acc);
      this.t += this.ratio;
    }
    const keep = Math.floor(this.t) - (half - 1);
    if (keep > 0) {
      buf.copyWithin(0, keep, this.len);
      this.len -= keep;
      this.t -= keep;
    }
    return true;
  }
}
registerProcessor("pcm-capture", PcmCapture);
`;

export type MicErrorKind = "denied" | "no-device" | "busy" | "unsupported" | "insecure" | "failed";

export class MicError extends Error {
  constructor(
    readonly kind: MicErrorKind,
    message: string,
  ) {
    super(message);
    this.name = "MicError";
  }
}

export const MIC_ERROR_TEXT: Record<MicErrorKind, string> = {
  denied: "Microphone access is blocked. Allow it for this site in your browser's settings, then try again.",
  "no-device": "No microphone was found. Connect one, then try again.",
  busy: "The microphone is in use by another app or couldn't be started.",
  unsupported: "Voice isn't supported in this browser. You can type instead.",
  insecure: "Microphone access needs a secure page (https or localhost). You can type instead.",
  failed: "Couldn't start the microphone.",
};

/** What this browser can do, checked before asking for anything. */
export function voiceSupport(): { ok: true } | { ok: false; kind: MicErrorKind } {
  if (typeof window === "undefined") return { ok: false, kind: "unsupported" };
  if (!window.isSecureContext) return { ok: false, kind: "insecure" };
  const ok =
    !!navigator.mediaDevices?.getUserMedia &&
    typeof AudioContext !== "undefined" &&
    typeof AudioWorkletNode !== "undefined" &&
    typeof WebSocket !== "undefined";
  return ok ? { ok: true } : { ok: false, kind: "unsupported" };
}

function classify(err: unknown): MicError {
  const name = err instanceof DOMException || err instanceof Error ? err.name : "";
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
    case "PermissionDeniedError":
      return new MicError("denied", MIC_ERROR_TEXT.denied);
    case "NotFoundError":
    case "OverconstrainedError":
    case "DevicesNotFoundError":
      return new MicError("no-device", MIC_ERROR_TEXT["no-device"]);
    case "NotReadableError":
    case "AbortError":
    case "TrackStartError":
      return new MicError("busy", MIC_ERROR_TEXT.busy);
    default:
      return new MicError("failed", MIC_ERROR_TEXT.failed);
  }
}

/** The mic's permission state without prompting, where the browser can say ("prompt" otherwise). */
export async function micPermission(): Promise<"granted" | "denied" | "prompt"> {
  try {
    const status = await navigator.permissions.query({ name: "microphone" as PermissionName });
    return status.state;
  } catch {
    return "prompt"; // Safari and others can't be asked
  }
}

export interface MicCapture {
  /** The microphone itself. */
  stream: MediaStream;
  /** What the browser VAD listens to: the microphone after the denoiser (the microphone itself without one). */
  vadStream: MediaStream;
  /** The user's voice, for the presence field. */
  analyser: AnalyserNode;
  /** RNNoise is running on the microphone (docs/DESIGN.md §3.10). */
  denoised: boolean;
  stop: () => void;
}

/**
 * Opens the microphone and streams PCM16 16 kHz frames to `onFrame` until `stop()`. Throws MicError. The frame
 * callback gets an ArrayBuffer it owns. With `denoise`, RNNoise (lib/voice/denoise.ts) cleans the audio before the
 * uplink, the analyser and the VAD's stream; if it can't run, the microphone is used as it is.
 *
 *   mic ─► [RNNoise] ─┬─► AudioWorklet (16 kHz PCM16) ─► onFrame
 *                     ├─► AnalyserNode (presence field)
 *                     └─► MediaStreamDestination (the browser VAD's stream)
 */
export async function openMic(
  ctx: AudioContext,
  onFrame: (pcm: ArrayBuffer) => void,
  /** The device went away or access was revoked while open (not called by `stop()`). */
  onEnded?: () => void,
  opts: { denoise?: boolean } = {},
): Promise<MicCapture> {
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
  } catch (err) {
    throw classify(err);
  }
  let denoiser: Denoiser | null = null;
  try {
    const url = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: "text/javascript" }));
    try {
      await ctx.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }
    if (opts.denoise) denoiser = await createDenoiser(ctx);
    const mic = ctx.createMediaStreamSource(stream);
    if (denoiser) mic.connect(denoiser.node);
    const source: AudioNode = denoiser ? denoiser.node : mic;
    const toVad = denoiser ? ctx.createMediaStreamDestination() : null;
    if (toVad) source.connect(toVad);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.3;
    const node = new AudioWorkletNode(ctx, "pcm-capture", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      processorOptions: { targetRate: INPUT_SAMPLE_RATE, frameSamples: UPLINK_FRAME_SAMPLES },
    });
    node.port.onmessage = (e: MessageEvent<ArrayBuffer>) => onFrame(e.data);
    if (onEnded) for (const t of stream.getAudioTracks()) t.addEventListener("ended", onEnded, { once: true });
    // Browsers only pull a node that leads to the destination; a zero-gain sink keeps it running silently.
    const sink = ctx.createGain();
    sink.gain.value = 0;
    source.connect(analyser);
    source.connect(node);
    node.connect(sink);
    sink.connect(ctx.destination);
    return {
      stream,
      vadStream: toVad ? toVad.stream : stream,
      analyser,
      denoised: denoiser !== null,
      stop: () => {
        node.port.onmessage = null;
        for (const t of stream.getTracks()) t.stop();
        try {
          mic.disconnect();
          node.disconnect();
          sink.disconnect();
          analyser.disconnect();
          toVad?.disconnect();
        } catch {
          // already disconnected
        }
        denoiser?.destroy();
      },
    };
  } catch (err) {
    denoiser?.destroy();
    for (const t of stream.getTracks()) t.stop();
    throw err instanceof MicError ? err : new MicError("failed", MIC_ERROR_TEXT.failed);
  }
}
