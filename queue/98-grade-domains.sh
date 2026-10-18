#!/usr/bin/env bash
exec .venv/bin/python behaviour_check.py grade samples/domains.jsonl --base runs/base-s2 --report reports/domains_samples.md
