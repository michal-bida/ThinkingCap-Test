"""Mix an arm with the 0.6B's own finished traces on general prompts (arm "M<arm>").

    python build_mix.py data/arms/S.jsonl --keep data/arms/S_keep.jsonl \
        --general samples/general.jsonl --out data/arms/MS.jsonl

General rows (data/general.jsonl, sampled with sample_traces.py) have no grader. Per
prompt, the shortest sample that closed its thinking, was not truncated and has no
repetition loop is used as is. It shows the adapter how open-ended work ends while the
arm teaches where reasoning can stop. The general qids are offset by 1,000,000.
"""

import argparse
import json
import random
import re
from collections import defaultdict

OFFSET = 1_000_000


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arm")
    parser.add_argument("--keep")
    parser.add_argument("--general", required=True)
    parser.add_argument("--data", default="data/general.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def looping(text, n=12, times=3):
    """True if some n-word span repeats at least `times` times."""
    words = text.split()
    counts = defaultdict(int)
    for i in range(len(words) - n + 1):
        key = tuple(words[i:i + n])
        counts[key] += 1
        if counts[key] >= times:
            return True
    return False


def main():
    arguments = parse_arguments()
    rows = [json.loads(line) for line in open(arguments.arm)]
    if arguments.keep:
        keep = {json.loads(line)["qid"] for line in open(arguments.keep)}
        rows = [row for row in rows if row["qid"] in keep]
    prompts = [json.loads(line) for line in open(arguments.data)]

    by_prompt = defaultdict(list)
    for trace in map(json.loads, open(arguments.general)):
        text = trace["text"]
        if trace["truncated"] or "</think>" not in text or looping(text):
            continue
        by_prompt[trace["qid"]].append(trace)
    general = []
    for qid, traces in sorted(by_prompt.items()):
        best = min(traces, key=lambda t: t["output_tokens"])
        thinking, answer = best["text"].split("</think>", 1)
        thinking = thinking.replace("<think>", "").strip()
        row = prompts[qid]
        general.append({"prompt": row["prompt"],
                        "completion": [{"role": "assistant",
                                        "content": f"<think>\n{thinking}\n</think>\n\n{answer.strip()}"}],
                        "qid": OFFSET + qid, "source": row["source"], "answer": "", "task": "open"})

    mixed = rows + general
    random.Random(arguments.seed).shuffle(mixed)
    with open(arguments.out, "w") as stream:
        for row in mixed:
            stream.write(json.dumps(row) + "\n")
    print(f"{len(rows)} arm rows + {len(general)} general rows "
          f"(of {len(prompts)} prompts) -> {arguments.out}")


if __name__ == "__main__":
    main()
