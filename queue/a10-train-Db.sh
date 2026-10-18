#!/usr/bin/env bash
# Db: D with balanced pairs (per domain, long pairs capped at the number of wrong pairs); else as D.
exec .venv/bin/python train_dpo_offline.py samples/domains.graded.jsonl --balance --out adapters/Db
