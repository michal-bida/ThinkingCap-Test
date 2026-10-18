#!/usr/bin/env bash
# Serve a GGUF model with llama.cpp for sample_llama.py, split between GPU and CPU.
#
#   ./serve_llama.sh ~/models/thinkingcap/ThinkingCap-Qwen3.8-27B-Q4_K_M.gguf [parallel] [tokens per slot]
#
# --fit (on by default) puts as many layers on the GPU as fit and leaves the rest on
# the CPU. Each of the `parallel` slots gets `tokens per slot` of context.
set -euo pipefail

LLAMA=${LLAMA:-$HOME/opt/llama.cpp/b11497}
MODEL=$1
PARALLEL=${2:-4}
PER_SLOT=${3:-8704}

export LD_LIBRARY_PATH="$LLAMA/cudart-llama-b11497-bin-ubuntu-cuda-12.8-x64:$LLAMA/llama-b11497${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$LLAMA/llama-b11497/llama-server" -m "$MODEL" \
    --host 127.0.0.1 --port 8080 \
    -np "$PARALLEL" -c $((PARALLEL * PER_SLOT)) \
    -t 12 --jinja --reasoning-format none --no-webui
