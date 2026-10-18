#!/usr/bin/env bash
.venv/bin/python evaluate.py --samples 2 --gpu-memory-utilization 0.8 --adapter adapters/G1 --out runs/G1-s2 \
  && .venv/bin/python compare.py runs/base-s2 runs/G1-s2 > reports/compare_G1.txt
