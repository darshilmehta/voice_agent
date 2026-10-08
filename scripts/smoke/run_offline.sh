#!/usr/bin/env bash
# Smoke test 12 — run every automated smoke test with the network OFF.
#
#   1. Turn Wi-Fi off (and unplug Ethernet).
#   2. scripts/smoke/run_offline.sh            (or a subset: ONLY="05_docling" scripts/smoke/run_offline.sh)
#   3. Turn Wi-Fi back on. Results: data/smoke/offline_run.log
#
# Needs: Ollama service and the Qdrant container running (both local).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="$ROOT/data/smoke/offline_run${ONLY:+_$(echo $ONLY | tr " " "_")}.log"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"

export UV_OFFLINE=1                # uv may only use its local cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WHISPER_VARIANTS="mlx-small,mlx-turbo,fw-small"   # skip fw-turbo (CPU, ~3 min, already measured)

{
  echo "=== offline smoke run $(date '+%Y-%m-%d %H:%M:%S') ==="
  if curl -s -m 4 -o /dev/null https://huggingface.co 2>/dev/null || curl -s -m 4 -o /dev/null https://1.1.1.1 2>/dev/null; then
    echo "NETWORK IS STILL UP: turn Wi-Fi off and run again. Aborting."; exit 2
  fi
  echo "network: unreachable ✓"
  echo "ollama: $(curl -s -m 3 http://127.0.0.1:11434/api/version || echo DOWN)"
  echo "qdrant: $(curl -s -m 3 http://127.0.0.1:6333/readyz || echo DOWN)"
} 2>&1 | tee "$LOG"
[ "${PIPESTATUS[0]}" = "2" ] && exit 2

summary=()
TESTS="${ONLY:-01_ollama 02_qdrant 03_bge_m3 04_reranker 05_docling 09_kokoro 06_whisper 08_vad 11_end_to_end}"
for t in $TESTS; do
  echo -e "\n\n######## $t ########" | tee -a "$LOG"
  start=$(date +%s)
  args=(); [ "$t" = "01_ollama" ] && args=(qwen3:4b-instruct)   # the configured model; 8b was compared online
  uv run --offline "scripts/smoke/$t.py" "${args[@]+"${args[@]}"}" >>"$LOG" 2>&1
  rc=$?
  line="$t: $([ $rc = 0 ] && echo PASS || echo "FAIL (exit $rc)") in $(( $(date +%s) - start ))s"
  echo "$line" | tee -a "$LOG"
  summary+=("$line")
done

{
  echo -e "\n\n=== summary ==="
  printf '%s\n' "${summary[@]}"
  echo "finished $(date '+%H:%M:%S'). Turn Wi-Fi back on."
} | tee -a "$LOG"
