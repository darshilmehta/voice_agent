/**
 * Noise suppression on the microphone (docs/DESIGN.md §3.10 "Noisy rooms"): RNNoise (xiph/rnnoise, a small recurrent
 * network trained on speech against steady and non-steady noise) in an AudioWorklet, from @sapphi-red/web-noise-
 * suppressor (pinned). It runs before everything else that hears the user: the browser VAD, the uplink to the server
 * (and so its VAD and Whisper), and the user's waves in the presence field. The browser's own echo cancellation, noise
 * suppression and gain control stay on in front of it.
 *
 * The worklet and its WASM are served by this app (`public/denoise/`, copied from node_modules by
 * scripts/copy-vad-assets.mjs): nothing comes from a CDN. RNNoise works on 48 kHz audio, so the voice page opens its
 * AudioContext at 48 kHz; at any other rate (a browser that refused it) the denoiser is skipped.
 *
 * Cost (scripts/eval/denoise.mjs, the same worklet and WASM in V8): ~28 µs of CPU per 128-sample render quantum
 * (2.67 ms of audio), about 1% of one core; it delays the audio by 512 samples (10.7 ms).
 */

const ASSET_BASE = "/denoise/";
export const DENOISE_SAMPLE_RATE = 48_000;

export interface Denoiser {
  node: AudioNode;
  destroy: () => void;
}

/** RNNoise as an AudioNode on `ctx`, or null when it can't run there (not 48 kHz, no AudioWorklet, assets missing). */
export async function createDenoiser(ctx: AudioContext): Promise<Denoiser | null> {
  if (ctx.sampleRate !== DENOISE_SAMPLE_RATE || typeof AudioWorkletNode === "undefined") return null;
  try {
    const { loadRnnoise, RnnoiseWorkletNode } = await import("@sapphi-red/web-noise-suppressor");
    const [wasmBinary] = await Promise.all([
      loadRnnoise({ url: `${ASSET_BASE}rnnoise.wasm`, simdUrl: `${ASSET_BASE}rnnoise_simd.wasm` }),
      ctx.audioWorklet.addModule(`${ASSET_BASE}rnnoise-worklet.js`),
    ]);
    const node = new RnnoiseWorkletNode(ctx, { wasmBinary, maxChannels: 1 });
    return {
      node,
      destroy: () => {
        try {
          node.disconnect();
        } catch {
          // already disconnected
        }
        node.destroy(); // frees the WASM state in the worklet
      },
    };
  } catch (err) {
    console.warn("Noise suppression unavailable; the microphone is used as it is:", err);
    return null;
  }
}
