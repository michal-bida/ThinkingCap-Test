#!/usr/bin/env bash
# D: DPO on pairs of the same stored samples as G1 (chosen shortest correct; rejected wrong/truncated or >=1.5x longer correct).
exec .venv/bin/python train_dpo_offline.py samples/domains.graded.jsonl --out adapters/D
