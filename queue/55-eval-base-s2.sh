#!/usr/bin/env bash
exec .venv/bin/python evaluate.py --samples 2 --gpu-memory-utilization 0.8 --out runs/base-s2
