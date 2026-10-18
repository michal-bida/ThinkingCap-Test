"""Sample Qwen3-0.6B's own reasoning traces on the training questions.

    python sample_traces.py --per-source 100 --out samples/pilot.jsonl

Questions are taken per source from a fixed shuffle of data/train.jsonl, so a
larger --per-source later is a superset of a smaller one and its file can be
resumed: rows already in --out are skipped. Sampling matches the evaluation
(thinking on, temperature 0.6, top-p 0.95, top-k 20) so the traces are what the
model actually produces when it is scored.

Every finished sample is written at once, one JSON line each:
    {"qid", "source", "task", "answer", "sample", "seed", "prompt_tokens",
     "text", "token_ids", "output_tokens", "thinking_tokens", "truncated", "correct"}
"""

import argparse
import json
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from utils.eval.recipe import SAMPLING, SEED
from utils.train.rewards import is_task_correct

MODEL = "Qwen/Qwen3-0.6B"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-source", type=int, required=True,
                        help="Questions taken from each source")
    parser.add_argument("--sources", help="Comma-separated sources to sample (default: all)")
    parser.add_argument("--samples", type=int, default=4, help="Traces per question")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def select(path, per_source, sources=None):
    """(qid, row) pairs: the first per_source of each source in a fixed shuffle."""
    by_source = defaultdict(list)
    for qid, line in enumerate(open(path)):
        row = json.loads(line)
        by_source[row["source"]].append((qid, row))
    chosen = []
    for source in sorted(sources or by_source):
        rows = by_source[source]
        random.Random(f"{source}-0").shuffle(rows)
        chosen += rows[:per_source]
    return chosen


def load_engine(arguments, max_model_len):
    # Same choices as evaluate.py: no FlashInfer kernels without nvcc.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM
    return LLM(model=MODEL, max_model_len=max_model_len,
               gpu_memory_utilization=arguments.gpu_memory_utilization,
               generation_config="vllm", seed=arguments.seed, disable_log_stats=True,
               attention_backend=None if shutil.which("nvcc") else "TRITON_ATTN")


def main():
    arguments = parse_arguments()
    from vllm import SamplingParams

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["qid"], r["sample"]) for r in map(json.loads, out.open())}

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    jobs = {}
    for qid, row in select(arguments.data, arguments.per_source,
                            arguments.sources.split(",") if arguments.sources else None):
        ids = tokenizer.apply_chat_template(row["prompt"], tokenize=True,
                                            add_generation_prompt=True, enable_thinking=True)
        for sample in range(arguments.samples):
            if (qid, sample) not in done:
                seed = arguments.seed + qid * 1000 + sample
                jobs[f"{qid}-{sample}"] = (qid, sample, seed, row, ids)
    print(f"{len(done)} samples already in {out}, {len(jobs)} to go", flush=True)
    if not jobs:
        return

    longest = max(len(job[4]) for job in jobs.values())
    engine = load_engine(arguments, longest + arguments.max_tokens).llm_engine
    for request_id, (qid, sample, seed, row, ids) in jobs.items():
        engine.add_request(request_id, {"prompt_token_ids": ids},
                           SamplingParams(max_tokens=arguments.max_tokens, seed=seed, **SAMPLING))

    finished, generated, start = 0, 0, time.time()
    with out.open("a") as stream:
        while engine.has_unfinished_requests():
            for output in engine.step():
                if not output.finished:
                    continue
                qid, sample, seed, row, ids = jobs[output.request_id]
                completion = output.outputs[0]
                token_ids = list(completion.token_ids)
                stream.write(json.dumps({
                    "qid": qid, "source": row["source"], "task": row["task"],
                    "answer": row["answer"], "sample": sample, "seed": seed,
                    "prompt_tokens": len(ids), "text": completion.text, "token_ids": token_ids,
                    "output_tokens": len(token_ids),
                    "thinking_tokens": token_ids.index(close_id) + 1 if close_id in token_ids else None,
                    "truncated": completion.finish_reason == "length",
                    "correct": (is_task_correct(completion.text, row["answer"], row["task"])
                                if row["task"] in ("math", "multiple_choice") else None),
                }) + "\n")
                stream.flush()
                finished += 1
                generated += len(token_ids)
                if finished % 50 == 0 or finished == len(jobs):
                    elapsed = time.time() - start
                    print(f"{finished}/{len(jobs)}  {generated / elapsed:.0f} tok/s  "
                          f"{elapsed / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
