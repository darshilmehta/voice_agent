/**
 * Voice analysis for the presence field (docs/DESIGN.md §3.8). Each voice (the agent's playback, the user's mic) is
 * read from an AnalyserNode every animation frame:
 *
 * - loudness: an RMS envelope with a fast attack (~40 ms) and slow release (~250 ms), so the field breathes instead
 *   of jittering;
 * - brightness: the spectral centroid, which tints the speaker's palette;
 * - onsets: a sudden energy rise (roughly a syllable) starts a ripple, a wave that travels outward for the agent and
 *   inward for the user.
 */

export const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));

export interface Meter {
  buf: Float32Array<ArrayBuffer>;
  freq: Uint8Array<ArrayBuffer>;
  /** Smoothed loudness, 0…1. */
  env: number;
  /** Smoothed brightness, 0…1. */
  tone: number;
  /** Slow-moving loudness, the baseline an onset rises above. */
  slow: number;
  lastOnset: number;
  /** Start times (seconds, same clock as the renderer's) of the four most recent ripples. */
  ripples: [number, number, number, number];
  ri: number;
}

export function createMeter(): Meter {
  return {
    buf: new Float32Array(1024),
    freq: new Uint8Array(512),
    env: 0,
    tone: 0,
    slow: 0,
    lastOnset: 0,
    ripples: [-100, -100, -100, -100],
    ri: 0,
  };
}

/**
 * Advance a meter by `dt` seconds at time `t`. Without an analyser (no session yet, or silence) the meter decays to 0.
 * `onsets` is false for reduced motion: waves are motion, the band still follows the voice.
 */
export function measure(
  an: AnalyserNode | null,
  m: Meter,
  dt: number,
  t: number,
  sampleRate: number,
  onsets: boolean,
): void {
  let level = 0;
  let tone = 0;
  if (an) {
    // The analyser's FFT size is fixed at creation (1024 time-domain samples, 512 bins).
    an.getFloatTimeDomainData(m.buf);
    let sum = 0;
    for (let i = 0; i < m.buf.length; i++) sum += m.buf[i] * m.buf[i];
    const db = 20 * Math.log10(Math.sqrt(sum / m.buf.length) + 1e-9);
    level = clamp((db + 58) / 40, 0, 1);
    if (level > 0.02) {
      an.getByteFrequencyData(m.freq);
      let num = 0;
      let den = 0;
      const binHz = sampleRate / 2 / m.freq.length;
      for (let i = 1; i < m.freq.length; i++) {
        num += i * binHz * m.freq[i];
        den += m.freq[i];
      }
      tone = den ? clamp((num / den - 400) / 2600, 0, 1) : 0;
    }
  }
  const k = level > m.env ? 1 - Math.exp(-dt / 0.04) : 1 - Math.exp(-dt / 0.25);
  m.env += (level - m.env) * k;
  m.tone += (tone - m.tone) * (1 - Math.exp(-dt / 0.15));
  m.slow += (level - m.slow) * (1 - Math.exp(-dt / 0.3));
  if (onsets && level > 0.35 && level - m.slow > 0.16 && t - m.lastOnset > 0.35) {
    m.lastOnset = t;
    m.ripples[m.ri++ % 4] = t;
  }
}
