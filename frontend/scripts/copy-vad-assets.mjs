// Copies the voice page's audio assets from node_modules into public/, so the app serves them itself and never reaches
// for a CDN (docs/DESIGN.md §1, §6: everything local):
//   public/vad/      the browser VAD (Silero v5 via @ricky0123/vad-web) and its ONNX runtime
//   public/denoise/  the denoiser (RNNoise via @sapphi-red/web-noise-suppressor: its AudioWorklet and WASM, §3.10)
// Both are git-ignored; this runs before `npm run dev` and `npm run build` (see package.json).

import { copyFileSync, existsSync, mkdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const FILES = [
  // [directory under public/, package directory under node_modules, file, name in public/ (default: the same)]
  ["vad", "@ricky0123/vad-web/dist", "bundle.min.js"], // window.vad.MicVAD
  ["vad", "@ricky0123/vad-web/dist", "vad.worklet.bundle.min.js"], // the VAD's AudioWorklet
  ["vad", "@ricky0123/vad-web/dist", "silero_vad_v5.onnx"], // the model
  ["vad", "@ricky0123/vad-web/dist", "bundle.min.js.LICENSE.txt"],
  ["vad", "onnxruntime-web/dist", "ort.wasm.min.js"], // window.ort (CPU/WASM only)
  ["vad", "onnxruntime-web/dist", "ort-wasm-simd-threaded.mjs"],
  ["vad", "onnxruntime-web/dist", "ort-wasm-simd-threaded.wasm"],
  ["denoise", "@sapphi-red/web-noise-suppressor/dist/rnnoise", "workletProcessor.js", "rnnoise-worklet.js"],
  ["denoise", "@sapphi-red/web-noise-suppressor/dist", "rnnoise.wasm"],
  ["denoise", "@sapphi-red/web-noise-suppressor/dist", "rnnoise_simd.wasm"],
  ["denoise", "@sapphi-red/web-noise-suppressor", "LICENSE", "LICENSE.txt"],
];

let copied = 0;
for (const [sub, dir, file, name = file] of FILES) {
  const from = join(root, "node_modules", dir, file);
  if (!existsSync(from)) {
    console.error(`copy-vad-assets: missing ${from}. Run "npm ci" first.`);
    process.exit(1);
  }
  const out = join(root, "public", sub);
  mkdirSync(out, { recursive: true });
  const to = join(out, name);
  if (existsSync(to) && statSync(to).size === statSync(from).size && statSync(to).mtimeMs >= statSync(from).mtimeMs) continue;
  copyFileSync(from, to);
  copied++;
}
console.log(`copy-vad-assets: ${copied ? `copied ${copied} file(s) to` : "up to date in"} public/vad, public/denoise`);
