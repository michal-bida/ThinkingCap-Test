#!/usr/bin/env bash
.venv/bin/python evaluate.py --samples 2 --gpu-memory-utilization 0.8 --adapter adapters/D --out runs/D-s2 \
  && .venv/bin/python compare.py runs/base-s2 runs/D-s2 > reports/compare_D.txt
