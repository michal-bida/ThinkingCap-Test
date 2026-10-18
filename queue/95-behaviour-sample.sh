#!/usr/bin/env bash
exec .venv/bin/python behaviour_check.py sample --per-domain 300 --out samples/behaviour.jsonl
