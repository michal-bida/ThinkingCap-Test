#!/usr/bin/env bash
exec .venv/bin/python train_sft.py data/arms/MS.jsonl --out adapters/MS
