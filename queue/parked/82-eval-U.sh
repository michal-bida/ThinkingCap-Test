#!/usr/bin/env bash
exec .venv/bin/python evaluate.py --samples 2 --gpu-memory-utilization 0.8 --adapter adapters/U --out runs/U-s2
