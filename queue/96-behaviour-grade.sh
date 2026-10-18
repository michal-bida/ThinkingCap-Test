#!/usr/bin/env bash
exec .venv/bin/python behaviour_check.py grade samples/behaviour.jsonl --base runs/base-s2
