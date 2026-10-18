#!/usr/bin/env bash
.venv/bin/python evaluate.py --samples 2 --gpu-memory-utilization 0.8 --adapter adapters/S2 --out runs/S2-s2 \
  && .venv/bin/python compare.py runs/base-s2 runs/S2-s2 > reports/compare_S2.txt
