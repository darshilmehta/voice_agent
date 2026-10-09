// Copies the browser VAD (Silero v5 via @ricky0123/vad-web) and its ONNX runtime from node_modules into
// public/vad/, so the app serves them itself and never reaches for a CDN (docs/DESIGN.md §1, §6: everything local).
// public/vad/ is git-ignored; this runs before `npm run dev` and `npm run build` (see package.json).

import { copyFileSync, existsSync, mkdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, "public", "vad");

const FILES = [
  // [package directory under node_modules, file]
  ["@ricky0123/vad-web/dist", "bundle.min.js"], // window.vad.MicVAD
  ["@ricky0123/vad-web/dist", "vad.worklet.bundle.min.js"], // the VAD's AudioWorklet
  ["@ricky0123/vad-web/dist", "silero_vad_v5.onnx"], // the model
  ["@ricky0123/vad-web/dist", "bundle.min.js.LICENSE.txt"],
  ["onnxruntime-web/dist", "ort.wasm.min.js"], // window.ort (CPU/WASM only)
  ["onnxruntime-web/dist", "ort-wasm-simd-threaded.mjs"],
  ["onnxruntime-web/dist", "ort-wasm-simd-threaded.wasm"],
];

mkdirSync(out, { recursive: true });
let copied = 0;
for (const [dir, file] of FILES) {
  const from = join(root, "node_modules", dir, file);
  if (!existsSync(from)) {
    console.error(`copy-vad-assets: missing ${from}. Run "npm ci" first.`);
    process.exit(1);
  }
  const to = join(out, file);
  if (existsSync(to) && statSync(to).size === statSync(from).size && statSync(to).mtimeMs >= statSync(from).mtimeMs) continue;
  copyFileSync(from, to);
  copied++;
}
console.log(`copy-vad-assets: ${copied ? `copied ${copied} file(s) to` : "up to date in"} public/vad`);
