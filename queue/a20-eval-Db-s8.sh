#!/usr/bin/env bash
# Full evaluation (8 samples, the leaderboard recipe). vLLM may OOM on its first batch at 0.8 on this
# card; retry once at 0.78 (only sizes the KV cache; sampling and seeds unchanged). A crashed attempt
# leaves an empty records.jsonl that evaluate.py refuses to overwrite, so clear it first.
clear_empty() { [ -s runs/Db-s8/records.jsonl ] || rm -rf runs/Db-s8; }
clear_empty
.venv/bin/python evaluate.py --gpu-memory-utilization 0.8 --adapter adapters/Db --out runs/Db-s8 \
  || { sleep 10; clear_empty; .venv/bin/python evaluate.py --gpu-memory-utilization 0.78 --adapter adapters/Db --out runs/Db-s8; }
