#!/usr/bin/env bash
exec .venv/bin/python sample_traces.py --data data/general.jsonl --per-source 1000 --samples 2 --gpu-memory-utilization 0.8 --out samples/general.jsonl
