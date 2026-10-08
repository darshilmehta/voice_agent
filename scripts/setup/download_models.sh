#!/usr/bin/env bash
# Download all Hugging Face model weights into the project-local cache.
# After this, the app runs with HF_HUB_OFFLINE=1 and never touches the network.
#
#   scripts/setup/download_models.sh [embeddings|whisper|all]
#
# Re-running is safe: completed files are skipped, partial files resume.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export HF_HOME="$ROOT/data/models/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$HF_HOME"

EMBEDDINGS=(
  "BAAI/bge-m3|onnx/*,imgs/*"            # dense+sparse embeddings; skip duplicate ONNX weights
  "BAAI/bge-reranker-v2-m3|"             # multilingual cross-encoder reranker
)
WHISPER=(
  "Systran/faster-whisper-small|"                  # CPU baseline
  "mobiuslabsgmbh/faster-whisper-large-v3-turbo|"  # CPU, better Hindi
  "mlx-community/whisper-small-mlx|"               # Apple GPU (MLX)
  "mlx-community/whisper-large-v3-turbo|"          # Apple GPU (MLX), better Hindi
)
TTS=(
  "hexgrad/Kokoro-82M|eval/*,samples/*"  # weights + all voice packs (en + hi included)
)

# Docling's layout / table / OCR models live outside the HF cache; docling-tools
# fetches them into a fixed directory that the pipeline points at via artifacts_path.
download_docling() {
  local out="$ROOT/data/models/docling"
  mkdir -p "$out"
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    echo "==> docling models (attempt $attempt)"
    if uvx --from docling docling-tools models download -o "$out" >/dev/null; then
      echo "    done: docling models"; du -sh "$out"; return 0
    fi
    sleep 15
  done
  echo "    FAILED: docling models" >&2; return 1
}

download() {
  local repo="${1%%|*}" exclude="${1#*|}" args=()
  [[ -n "$exclude" ]] && IFS=',' read -ra pats <<<"$exclude" && for p in "${pats[@]}"; do args+=(--exclude "$p"); done
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    echo "==> $repo (attempt $attempt)"
    if uvx --from huggingface_hub hf download "$repo" ${args[@]+"${args[@]}"} >/dev/null; then
      echo "    done: $repo"; return 0
    fi
    sleep 15
  done
  echo "    FAILED: $repo" >&2; return 1
}

WITH_DOCLING=0
case "${1:-all}" in
  embeddings) set -- "${EMBEDDINGS[@]}" ;;
  whisper)    set -- "${WHISPER[@]}" ;;
  tts)        set -- "${TTS[@]}" ;;
  docling)    download_docling; exit ;;
  all)        set -- "${EMBEDDINGS[@]}" "${WHISPER[@]}" "${TTS[@]}"; WITH_DOCLING=1 ;;
  *) echo "usage: $0 [embeddings|whisper|tts|docling|all]" >&2; exit 2 ;;
esac

for entry in "$@"; do download "$entry"; done
if (( WITH_DOCLING )); then download_docling; fi
du -sh "$HF_HOME/hub"/models--* 2>/dev/null
