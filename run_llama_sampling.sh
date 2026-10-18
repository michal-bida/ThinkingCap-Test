#!/usr/bin/env bash
# Sample ThinkingCap-Qwen3.8-27B, then base Qwen3.8-27B at its default (xhigh) and
# low reasoning effort, on the same pilot questions. One model on the GPU at a time.
#
#   timeout 46h ./run_llama_sampling.sh [thinkingcap] [qwen3.8] [qwen3.8-low]
#
# With no names it runs all three.
# Every run resumes from its samples/llama-<name>.jsonl, so rerunning continues.
set -uo pipefail
cd "$(dirname "$0")"

MODELS=$HOME/models
COMMON=(--qids samples/pilot.jsonl --limit 300 --temperature 1.0 --max-tokens 8192 --parallel 4)
server=

stop_server() {
    if [ -n "$server" ]; then
        kill "$server" 2>/dev/null
        wait "$server" 2>/dev/null
        server=
    fi
}
trap 'stop_server; exit 143' TERM INT

run() {
    local name=$1 model=$2
    shift 2
    for attempt in 1 2 3; do
        echo "$(date '+%F %T') $name: attempt $attempt"
        ./serve_llama.sh "$model" 4 9216 >> "logs/llama-$name.log" 2>&1 &
        server=$!
        until curl -sf http://127.0.0.1:8080/health >/dev/null; do
            kill -0 "$server" 2>/dev/null || break
            sleep 5
        done
        .venv/bin/python sample_llama.py --model-name "$name" "${COMMON[@]}" "$@" \
            --out "samples/llama-$name.jsonl" >> "logs/sample-$name.log" 2>&1
        status=$?
        stop_server
        if [ $status -eq 0 ]; then
            echo "$(date '+%F %T') $name: done"
            return 0
        fi
        echo "$(date '+%F %T') $name: sampler exited with $status"
    done
    return 1
}

for name in "${@:-thinkingcap qwen3.8 qwen3.8-low}"; do
    for name in $name; do
        case $name in
            thinkingcap) run thinkingcap "$MODELS/thinkingcap/ThinkingCap-Qwen3.8-27B-Q4_K_M.gguf" ;;
            qwen3.8) run qwen3.8 "$MODELS/qwen3.8/Qwen3.8-27B-Q4_K_M.gguf" ;;
            qwen3.8-low) run qwen3.8-low "$MODELS/qwen3.8/Qwen3.8-27B-Q4_K_M.gguf" --reasoning-effort low ;;
            *) echo "unknown model $name"; exit 2 ;;
        esac
    done
done
echo "$(date '+%F %T') all done"
