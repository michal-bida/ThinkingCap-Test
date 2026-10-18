#!/usr/bin/env bash
# 4 own samples per domain prompt (11,327 prompts): SFT selection, DPO pairs, GRPO baseline.
exec .venv/bin/python behaviour_check.py sample --per-domain 0 --samples 4 --gpu-memory-utilization 0.8 --out samples/domains.jsonl
