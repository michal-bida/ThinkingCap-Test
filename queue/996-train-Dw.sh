#!/usr/bin/env bash
# Dw: D's ablation, rejected = wrong samples only (same settings as D otherwise).
exec .venv/bin/python train_dpo_offline.py samples/domains.graded.jsonl --kinds wrong --out adapters/Dw
