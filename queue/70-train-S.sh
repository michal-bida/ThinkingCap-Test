#!/usr/bin/env bash
exec .venv/bin/python train_sft.py data/arms/S.jsonl --keep data/arms/S_keep.jsonl --out adapters/S
