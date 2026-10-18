#!/usr/bin/env bash
# G1: one offline GRPO round on the stored base samples (4 per prompt), Kimi length reward.
exec .venv/bin/python train_grpo_offline.py samples/domains.graded.jsonl --out adapters/G1
