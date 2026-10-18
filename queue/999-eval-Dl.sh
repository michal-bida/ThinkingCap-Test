#!/usr/bin/env bash
# vLLM sometimes OOMs on its first batch at 0.8 on this card (marginal headroom); retry once at 0.78.
# Memory utilisation only sizes the KV cache; sampling settings and seeds are unchanged.
# A crashed attempt leaves an empty records.jsonl, which evaluate.py refuses to overwrite: clear it.
clear_empty() { [ -s runs/Dl-s2/records.jsonl ] || rm -rf runs/Dl-s2; }
clear_empty
{ .venv/bin/python evaluate.py --samples 2 --gpu-memory-utilization 0.8 --adapter adapters/Dl --out runs/Dl-s2 \
  || { sleep 10; clear_empty; .venv/bin/python evaluate.py --samples 2 --gpu-memory-utilization 0.78 --adapter adapters/Dl --out runs/Dl-s2; }; } \
  && .venv/bin/python compare.py runs/base-s2 runs/Dl-s2 > reports/compare_Dl.txt
