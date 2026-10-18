#!/usr/bin/env bash
# Full evaluation (8 samples, the leaderboard recipe). vLLM may OOM on its first batch at 0.8 on this
# card; retry once at 0.78 (only sizes the KV cache; sampling and seeds unchanged). A crashed attempt
# leaves an empty records.jsonl that evaluate.py refuses to overwrite, so clear it first.
clear_empty() { [ -s runs/base-s8/records.jsonl ] || rm -rf runs/base-s8; }
clear_empty
.venv/bin/python evaluate.py --gpu-memory-utilization 0.8  --out runs/base-s8 \
  || { sleep 10; clear_empty; .venv/bin/python evaluate.py --gpu-memory-utilization 0.78  --out runs/base-s8; }
[ $? -eq 0 ] && [ -f runs/Db-s8/results.json ] && .venv/bin/python compare.py runs/base-s8 runs/Db-s8 > reports/compare_Db-s8.txt
