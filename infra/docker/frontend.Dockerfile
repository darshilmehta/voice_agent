# Frontend image. Build from the repo root:
#   docker build -f infra/docker/frontend.Dockerfile -t voice-agent-frontend .
# BACKEND_URL is read at request time, so the same image works against any backend.
FROM node:24-slim AS deps
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

FROM node:24-slim AS build
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules ./node_modules
COPY frontend/ ./
# `npm run build` first runs `prebuild` (scripts/copy-vad-assets.mjs), which copies the browser VAD model and the ONNX
# runtime from node_modules into public/vad/, and the denoiser (RNNoise worklet + WASM) into public/denoise/. Fail the
# build here if that didn't happen: without them the browser cannot do barge-in detection or noise suppression and
# every /vad/* or /denoise/* request would 404.
RUN npm run build \
 && test -s public/vad/silero_vad_v5.onnx \
 && test -s public/vad/ort-wasm-simd-threaded.wasm \
 && test -s public/denoise/rnnoise-worklet.js \
 && test -s public/denoise/rnnoise_simd.wasm

FROM node:24-slim
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3000 \
    BACKEND_URL=http://localhost:8000
# Standalone output holds server.js and the traced node_modules only; static assets and public/ are copied beside it.
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static
COPY --from=build --chown=node:node /app/public ./public
USER node
EXPOSE 3000
CMD ["node", "server.js"]
