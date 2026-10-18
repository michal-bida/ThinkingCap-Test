#!/usr/bin/env bash
# S2: each domain prompt's shortest correct own sample (build_s2.py), same LoRA SFT settings as S/MS.
exec .venv/bin/python train_sft.py data/arms/S2.jsonl --out adapters/S2
