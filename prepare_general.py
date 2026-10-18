"""General instruction prompts for keeping open-ended behaviour during SFT (not eval data).

    python prepare_general.py --out data/general.jsonl

Takes open-ended prompts from Dolly-15k (QA, brainstorming, writing, classification,
summarisation) and coding prompts from CodeAlpaca-20k, and writes them in the format of
data/train.jsonl with task "open" (no grader). Every prompt sharing an 8-gram with any
evaluation question (questions/questions.jsonl) is dropped, as a contamination guard.
"""

import argparse
import json
import random
import re

from datasets import load_dataset

DOLLY_CATEGORIES = {"open_qa", "general_qa", "brainstorming", "creative_writing",
                    "classification", "summarization", "information_extraction"}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dolly", type=int, default=1000)
    parser.add_argument("--code", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def grams(text, n=8):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def main():
    arguments = parse_arguments()
    rng = random.Random(arguments.seed)
    eval_grams = set()
    for line in open("questions/questions.jsonl"):
        for message in json.loads(line)["messages"]:
            eval_grams |= grams(message["content"])

    rows = []
    dolly = [r for r in load_dataset("databricks/databricks-dolly-15k", split="train")
             if r["category"] in DOLLY_CATEGORIES and len(r["instruction"] + r["context"]) < 1500]
    rng.shuffle(dolly)
    for r in dolly:
        prompt = r["instruction"] + (f"\n\n{r['context']}" if r["context"] else "")
        rows.append(("dolly", prompt))
    code = [r for r in load_dataset("sahil2801/CodeAlpaca-20k", split="train")
            if len(r["instruction"] + r["input"]) < 1000]
    rng.shuffle(code)
    for r in code:
        prompt = r["instruction"] + (f"\n\n{r['input']}" if r["input"] else "")
        rows.append(("codealpaca", prompt))

    taken, dropped = {"dolly": 0, "codealpaca": 0}, 0
    limits = {"dolly": arguments.dolly, "codealpaca": arguments.code}
    with open(arguments.out, "w") as stream:
        for source, prompt in rows:
            if taken[source] >= limits[source]:
                continue
            if grams(prompt) & eval_grams:
                dropped += 1
                continue
            taken[source] += 1
            stream.write(json.dumps({"prompt": [{"role": "user", "content": prompt}],
                                     "answer": "", "task": "open", "source": source}) + "\n")
    print(f"wrote {taken} to {arguments.out}; dropped {dropped} overlapping an eval question")


if __name__ == "__main__":
    main()
