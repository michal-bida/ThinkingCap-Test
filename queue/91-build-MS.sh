#!/usr/bin/env bash
exec .venv/bin/python build_mix.py data/arms/S.jsonl --keep data/arms/S_keep.jsonl --general samples/general.jsonl --out data/arms/MS.jsonl
