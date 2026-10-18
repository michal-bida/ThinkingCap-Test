"""Fidelity filter for edited training traces: does the edited thinking still carry the answer?

    python fidelity_check.py data/arms/T.jsonl --out data/arms/T_keep.jsonl

For each arm row, Qwen3-0.6B gets the question and the row's fixed <think>...</think>
block and generates only the answer, --samples times. Rows answered correctly at least
--min-correct times are written to --out ({qid, source, correct}), for train_sft.py --keep.
This filters training data; it is not an accuracy result.
"""

import argparse
import json
import os
import shutil
from collections import defaultdict

from transformers import AutoTokenizer

from analyze_traces import lenient_correct, thinking_of
from utils.eval.recipe import SAMPLING, SEED

MODEL = "Qwen/Qwen3-0.6B"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arm")
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--min-correct", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024, help="For the answer only")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(line) for line in open(arguments.arm)]
    jobs = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=True)
        thinking = thinking_of(row["completion"][0]["content"]).strip()
        ids = tokenizer(prompt + "<think>\n" + thinking + "\n</think>\n\n",
                        add_special_tokens=False)["input_ids"]
        for sample in range(arguments.samples):
            jobs.append((row, ids, SEED + row["qid"] * 1000 + 500 + sample))

    llm = LLM(model=MODEL, max_model_len=max(len(j[1]) for j in jobs) + arguments.max_tokens,
              gpu_memory_utilization=arguments.gpu_memory_utilization,
              generation_config="vllm", seed=SEED, disable_log_stats=True,
              attention_backend=None if shutil.which("nvcc") else "TRITON_ATTN")
    outputs = llm.generate([{"prompt_token_ids": ids} for _, ids, _ in jobs],
                           [SamplingParams(max_tokens=arguments.max_tokens, seed=seed, **SAMPLING)
                            for _, _, seed in jobs])

    correct = defaultdict(int)
    for (row, _, _), output in zip(jobs, outputs):
        graded = {"text": "</think>\n\n" + output.outputs[0].text, "answer": row["answer"],
                  "task": row["task"], "correct": False,
                  "truncated": output.outputs[0].finish_reason == "length"}
        correct[row["qid"]] += lenient_correct(graded)

    by_source = defaultdict(lambda: [0, 0])
    with open(arguments.out, "w") as stream:
        for row in rows:
            passed = correct[row["qid"]] >= arguments.min_correct
            by_source[row["source"]][0] += passed
            by_source[row["source"]][1] += 1
            if passed:
                stream.write(json.dumps({"qid": row["qid"], "source": row["source"],
                                         "correct": correct[row["qid"]]}) + "\n")
    for source, (passed, total) in sorted(by_source.items()):
        print(f"{source:15s} kept {passed}/{total} ({passed / total:.0%})")


if __name__ == "__main__":
    main()
