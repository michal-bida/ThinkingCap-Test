"""Arm S2: each domain prompt's shortest correct own sample, verbatim.

    python build_s2.py samples/domains.graded.jsonl --out data/arms/S2.jsonl

Samples come from behaviour_check.py (4 per prompt on data/domains, graded with the
evaluation's graders; KodCode's tests for code). Per prompt, the shortest sample that is
correct and not truncated is kept exactly as the model wrote it: selection only, no edits.
Prompts with no correct sample are left out. qids are offset per domain to stay unique.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

DOMAINS = ("code", "tools", "ifeval", "math")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples")
    parser.add_argument("--data", default="data/domains")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    best = {}
    for line in open(arguments.samples):
        record = json.loads(line)
        if not record.get("correct") or record["truncated"]:
            continue
        key = (record["task"], record["qid"])
        if key not in best or record["output_tokens"] < best[key]["output_tokens"]:
            best[key] = record
    rows, counts = [], Counter()
    for domain in DOMAINS:
        for line in open(Path(arguments.data) / f"{domain}.jsonl"):
            row = json.loads(line)
            record = best.get((domain, row["qid"]))
            if record is None:
                continue
            rows.append({"prompt": row["prompt"],
                         "completion": [{"role": "assistant", "content": record["text"]}],
                         "qid": DOMAINS.index(domain) * 100_000 + row["qid"], "source": row["source"],
                         "answer": row["answer"], "task": domain,
                         "output_tokens": record["output_tokens"]})
            counts[domain] += 1
    with open(arguments.out, "w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    print(f"{len(rows)} rows -> {arguments.out}: {dict(counts)}")


if __name__ == "__main__":
    main()
