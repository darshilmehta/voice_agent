// Runs the browser's denoiser (RNNoise from @sapphi-red/web-noise-suppressor: the exact AudioWorklet processor and
// WASM the voice page loads, frontend/lib/voice/denoise.ts) over audio files, offline, in Node. Used by
// scripts/eval/noise_robustness.py to measure what the denoiser does to noisy speech and how much CPU it costs.
//
//   node scripts/eval/denoise.mjs in1.f32 out1.f32 [in2.f32 out2.f32 …]
//
// Files are raw float32 little-endian mono at 48 kHz (RNNoise's rate; the voice page runs its AudioContext at 48 kHz).
// Prints one JSON line: {"quanta": render quanta processed, "audio_s", "cpu_ms", "realtime_factor", "us_per_quantum"}.
// Nothing is played and nothing leaves the machine. Needs `npm ci` in frontend/ first.

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const pkg = join(root, "frontend/node_modules/@sapphi-red/web-noise-suppressor/dist");
const QUANTUM = 128; // an AudioWorklet render quantum

// The AudioWorkletGlobalScope the processor expects.
let Processor = null;
globalThis.sampleRate = 48000;
globalThis.AudioWorkletProcessor = class {
  constructor() {
    this.port = { addEventListener() {}, removeEventListener() {}, postMessage() {}, start() {}, onmessage: null };
  }
};
globalThis.registerProcessor = (_name, cls) => {
  Processor = cls;
};
// Emscripten sees Node and wants a CommonJS __dirname; it never reads a file here (the WASM binary is passed in).
globalThis.__dirname = join(pkg, "rnnoise");
await import(pathToFileURL(join(pkg, "rnnoise/workletProcessor.js")).href);
if (!Processor) throw new Error("the RNNoise worklet didn't register a processor");

// The SIMD build, as the page picks when the browser has WASM SIMD (every browser the app supports, and Node 22).
const wasm = readFileSync(join(pkg, "rnnoise_simd.wasm"));
const wasmBinary = wasm.buffer.slice(wasm.byteOffset, wasm.byteOffset + wasm.byteLength);

async function newProcessor() {
  const p = new Processor({ processorOptions: { wasmBinary, maxChannels: 1 } });
  for (let i = 0; i < 400 && !p.processor; i++) await new Promise((r) => setTimeout(r, 5));
  if (!p.processor) throw new Error("the RNNoise WASM didn't load");
  return p;
}

const args = process.argv.slice(2);
if (args.length === 0 || args.length % 2) {
  console.error("usage: node scripts/eval/denoise.mjs in.f32 out.f32 [in.f32 out.f32 …]");
  process.exit(2);
}
let quanta = 0;
let cpuNs = 0n;
for (let i = 0; i < args.length; i += 2) {
  const buf = readFileSync(args[i]);
  const input = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + Math.floor(buf.byteLength / 4) * 4));
  const output = new Float32Array(input.length);
  const p = await newProcessor(); // a fresh state per file, as a fresh page has
  const inBlock = new Float32Array(QUANTUM);
  const outBlock = new Float32Array(QUANTUM);
  for (let at = 0; at + QUANTUM <= input.length; at += QUANTUM) {
    inBlock.set(input.subarray(at, at + QUANTUM));
    const t0 = process.hrtime.bigint();
    p.process([[inBlock]], [[outBlock]], {});
    cpuNs += process.hrtime.bigint() - t0;
    output.set(outBlock, at);
    quanta++;
  }
  p.destroy();
  writeFileSync(args[i + 1], Buffer.from(output.buffer));
}
const audioS = (quanta * QUANTUM) / 48000;
const cpuMs = Number(cpuNs) / 1e6;
console.log(
  JSON.stringify({
    quanta,
    audio_s: +audioS.toFixed(3),
    cpu_ms: +cpuMs.toFixed(1),
    realtime_factor: +(cpuMs / 1000 / audioS).toFixed(5),
    us_per_quantum: +((cpuMs * 1000) / quanta).toFixed(2),
  }),
);
