#!/usr/bin/env bash
# Run queue/NN-name.sh steps one at a time (one GPU job at a time).
#
#   ./run_queue.sh                      # no deadline: runs until stopped
#   ./run_queue.sh "2026-10-10 12:00"   # stop at the deadline, killing the running step
#
# Steps run in name order; each leaves queue/NN-name.done or .failed. New steps can be
# added while it runs; with nothing to do it checks again every minute.
set -uo pipefail
cd "$(dirname "$0")"
deadline=${1:+$(date -d "$1" +%s)}

cleanup_gpu() {
    pkill -x llama-server 2>/dev/null
    pkill -f "VLLM::EngineCore" 2>/dev/null
    sleep 5
}

while [ -z "$deadline" ] || [ "$(date +%s)" -lt "$deadline" ]; do
    step=
    for candidate in queue/*.sh; do
        [ -e "$candidate" ] || continue
        [ -e "${candidate%.sh}.done" ] || [ -e "${candidate%.sh}.failed" ] || { step=$candidate; break; }
    done
    if [ -z "$step" ]; then
        sleep 60
        continue
    fi
    log="logs/$(basename "${step%.sh}").log"
    if [ -n "$deadline" ]; then
        left=$(( deadline - $(date +%s) ))
        echo "$(date '+%F %T') start $step (${left}s left)"
        timeout --signal=TERM --kill-after=60 "$left" bash "$step" >> "$log" 2>&1
    else
        echo "$(date '+%F %T') start $step"
        bash "$step" >> "$log" 2>&1
    fi
    status=$?
    cleanup_gpu
    if [ $status -eq 0 ]; then touch "${step%.sh}.done"; else echo "$status" > "${step%.sh}.failed"; fi
    echo "$(date '+%F %T') end $step: exit $status"
done
cleanup_gpu
echo "$(date '+%F %T') deadline reached"
