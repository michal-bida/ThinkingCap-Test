#!/usr/bin/env bash
exec .venv/bin/python train_sft.py data/arms/V.jsonl --keep data/arms/T_keep.jsonl --out adapters/V
