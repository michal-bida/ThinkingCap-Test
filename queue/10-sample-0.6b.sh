# Qwen3-0.6B traces for training: 1000 questions each from GSM8K, ARC-Easy and CommonsenseQA, 4 samples.
exec .venv/bin/python sample_traces.py --sources arc_easy,commonsense_qa,gsm8k --per-source 1000 \
    --samples 4 --gpu-memory-utilization 0.85 --out samples/train3k.jsonl
