#!/usr/bin/env bash
# Dl: D's ablation, rejected = long samples only (same settings as D otherwise).
exec .venv/bin/python train_dpo_offline.py samples/domains.graded.jsonl --kinds long --out adapters/Dl
