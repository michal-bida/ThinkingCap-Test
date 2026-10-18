#!/usr/bin/env bash
exec .venv/bin/python train_sft.py data/arms/T.jsonl --keep data/arms/T_keep.jsonl --out adapters/T
