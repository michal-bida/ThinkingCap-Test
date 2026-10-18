"""Is a shortened thinking still enough for Qwen3-0.6B to answer? (stage 1 sufficiency test)

    python forced_answer.py samples/pilot.jsonl --out samples/forced_pilot.jsonl

For every leniently correct trace, the model is given the question and a fixed
<think>...</think> block and only generates the answer after it, --samples times:

    full   the trace's own thinking
    cut    thinking up to the end of the sentence where the gold answer first appears
    empty  no thinking at all (<think>\\n\\n</think>), the floor

Prints, per source, the answer accuracy for each variant and the thinking tokens it keeps.
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from statistics import mean

from transformers import AutoTokenizer

from analyze_traces import first_gold_end, lenient_correct, thinking_of
from utils.eval.recipe import SAMPLING, SEED

MODEL = "Qwen/Qwen3-0.6B"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512, help="For the answer only")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args()


def variants(record, question):
    thinking = thinking_of(record["text"]).strip()
    cut = first_gold_end(record, question)
    out = {"full": thinking, "empty": ""}
    if cut is not None:
        out["cut"] = thinking_of(record["text"])[:cut].strip()
    return out


def main():
    arguments = parse_arguments()
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows = {qid: json.loads(line) for qid, line in enumerate(open(arguments.data))}
    traces = [json.loads(line) for line in open(arguments.traces)]
    traces = [t for t in traces if lenient_correct(t)]

    jobs = []
    for index, trace in enumerate(traces):
        row = rows[trace["qid"]]
        prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=True)
        # The template opens <think>; close it ourselves around the given thinking.
        if not prompt.endswith("<think>\n"):
            prompt += "<think>\n"
        for name, thinking in variants(trace, row["prompt"][0]["content"]).items():
            text = prompt + (thinking + "\n" if thinking else "\n") + "</think>\n\n"
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            think_tokens = len(tokenizer(thinking, add_special_tokens=False)["input_ids"])
            for sample in range(arguments.samples):
                jobs.append({"trace": index, "variant": name, "ids": ids, "sample": sample,
                             "think_tokens": think_tokens,
                             "seed": SEED + trace["qid"] * 1000 + trace["sample"] * 10 + sample})
    print(f"{len(traces)} correct traces, {len(jobs)} answers to generate", flush=True)

    longest = max(len(job["ids"]) for job in jobs)
    llm = LLM(model=MODEL, max_model_len=longest + arguments.max_tokens,
              gpu_memory_utilization=arguments.gpu_memory_utilization,
              generation_config="vllm", seed=SEED, disable_log_stats=True,
              attention_backend=None if shutil.which("nvcc") else "TRITON_ATTN")
    params = [SamplingParams(max_tokens=arguments.max_tokens, seed=job["seed"], **SAMPLING)
              for job in jobs]
    outputs = llm.generate([{"prompt_token_ids": job["ids"]} for job in jobs], params)

    results = defaultdict(lambda: defaultdict(list))
    with open(arguments.out, "w") as stream:
        for job, output in zip(jobs, outputs):
            trace = traces[job["trace"]]
            answer = output.outputs[0].text
            # Grade the forced answer as a whole reply: thinking closed, then the answer.
            graded = dict(trace, text="</think>\n\n" + answer,
                          truncated=output.outputs[0].finish_reason == "length", correct=False)
            correct = lenient_correct(graded)
            stream.write(json.dumps({"qid": trace["qid"], "source": trace["source"],
                                     "trace_sample": trace["sample"], "variant": job["variant"],
                                     "sample": job["sample"], "think_tokens": job["think_tokens"],
                                     "answer_tokens": len(output.outputs[0].token_ids),
                                     "correct": correct, "answer": answer}) + "\n")
            results[trace["source"]][job["variant"]].append((correct, job["think_tokens"]))

    print(f"\n{'source':15s} {'variant':6s} {'n':>6s} {'accuracy':>9s} {'think tokens':>13s}")
    for source in sorted(results):
        for name in ("full", "cut", "empty"):
            values = results[source].get(name)
            if values:
                print(f"{source:15s} {name:6s} {len(values):6d} {mean(c for c, _ in values):9.1%} "
                      f"{mean(t for _, t in values):13.0f}")


if __name__ == "__main__":
    main()
