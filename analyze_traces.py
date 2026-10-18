"""Where do the tokens in sampled traces go? Stage 1 statistics for a sample_traces.py file.

    python analyze_traces.py samples/pilot.jsonl

Per source: length, strict and lenient accuracy, truncation, difficulty spread, and,
for correct traces, how far into the thinking the gold answer first appears and how
many "Wait / check" restarts follow it. Lenient grading accepts \\boxed{x} and
"answer is x" endings, to tell format failures apart from wrong reasoning.
"""

import json
import re
import sys
from collections import defaultdict
from statistics import mean, median

from transformers import AutoTokenizer

from utils.train.rewards import clean_answer, same_choice, same_number

BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
ANSWER_IS = re.compile(r"(?:final answer|answer is|answer)\s*:?\s*\**\s*\(?([^\n]+)", re.IGNORECASE)
CHECK = re.compile(r"\b(?:Wait|But wait|Let me (?:double-?check|check|verify|confirm)|"
                   r"Alternatively|Hmm|Hold on|Actually)\b")


def lenient_correct(record):
    if record["correct"]:
        return True
    if record["truncated"] or "</think>" not in record["text"]:
        return False
    answer = record["text"].split("</think>")[-1].replace("**", "")
    candidates = BOXED.findall(answer)[-1:] + ANSWER_IS.findall(answer)[-1:]
    for candidate in candidates:
        candidate = clean_answer(candidate.strip())
        if record["task"] == "math" and same_number(candidate, record["answer"]):
            return True
        if record["task"] == "multiple_choice" and same_choice(candidate, record["answer"]):
            return True
    return False


def gold_pattern(record, question=""):
    """Where the thinking first commits to the gold answer.

    Math: the number itself, or, when the question already contains it, only after
    "=", "is" or "get". Multiple choice: only an explicit "answer is C" style statement,
    since the model lists every option label while reading the question.
    """
    gold = re.escape(record["answer"])
    if record["task"] == "math":
        number = rf"(?<![\d.]){gold}(?![\d]|\.\d)"
        if re.search(number, question):
            return re.compile(rf"(?:=|\bis|\bget|\bgets|\bequals)\s*\$?\s*{number}")
        return re.compile(number)
    return re.compile(rf"(?i:answer|option|choice)\s*(?:is|would be|should be|must be|:)\s*"
                      rf"\**\(?{gold}\b")


def first_gold_end(record, question=""):
    """Character offset just past the sentence where the gold answer first appears, or None."""
    thinking = thinking_of(record["text"])
    match = gold_pattern(record, question).search(thinking)
    if not match:
        return None
    boundary = re.compile(r"[.!?](?:\s|$)|\n").search(thinking, match.end())
    return boundary.end() if boundary else len(thinking)


def thinking_of(text):
    return text.split("</think>")[0].replace("<think>", "")


def main(path):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    records = [json.loads(line) for line in open(path)]
    questions = {qid: json.loads(line)["prompt"][0]["content"]
                 for qid, line in enumerate(open("data/train.jsonl"))}
    by_source = defaultdict(list)
    for record in records:
        record["lenient"] = lenient_correct(record)
        by_source[record["source"]].append(record)

    print(f"{len(records)} traces from {path}\n")
    print(f"{'source':15s} {'n':>5s} {'tokens':>7s} {'median':>7s} {'think%':>7s} "
          f"{'strict':>7s} {'lenient':>8s} {'trunc':>6s}")
    for source, rows in sorted(by_source.items()):
        finished = [r for r in rows if r["thinking_tokens"]]
        think_share = mean(r["thinking_tokens"] / r["output_tokens"] for r in finished) if finished else 0
        print(f"{source:15s} {len(rows):5d} {mean(r['output_tokens'] for r in rows):7.0f} "
              f"{median(r['output_tokens'] for r in rows):7.0f} {think_share:7.0%} "
              f"{mean(r['correct'] for r in rows):7.0%} {mean(r['lenient'] for r in rows):8.0%} "
              f"{mean(r['truncated'] for r in rows):6.0%}")

    print("\nDifficulty: questions by number of leniently correct samples out of 4")
    for source, rows in sorted(by_source.items()):
        per_question = defaultdict(int)
        for r in rows:
            per_question[r["qid"]] += r["lenient"]
        counts = [sum(1 for v in per_question.values() if v == k) for k in range(5)]
        print(f"  {source:15s} " + "  ".join(f"{k}/4:{c:3d}" for k, c in enumerate(counts)))

    print("\nCorrect traces (lenient): where the gold answer first appears in the thinking")
    print(f"{'source':15s} {'n':>5s} {'found':>6s} {'first@':>7s} {'after':>7s} "
          f"{'checks':>7s} {'shortest/mean':>14s}")
    for source, rows in sorted(by_source.items()):
        correct = [r for r in rows if r["lenient"]]
        positions, tail_tokens, checks_after = [], [], []
        for r in correct:
            thinking = thinking_of(r["text"])
            match = gold_pattern(r, questions[r["qid"]]).search(thinking)
            if not match:
                continue
            # Token position of the first gold mention, as a share of the thinking.
            before = len(tokenizer(thinking[:match.end()])["input_ids"])
            total = len(tokenizer(thinking)["input_ids"])
            positions.append(before / total)
            tail_tokens.append(total - before)
            checks_after.append(len(CHECK.findall(thinking[match.end():])))
        # Shortest correct sample per question vs the mean correct sample.
        per_question = defaultdict(list)
        for r in correct:
            per_question[r["qid"]].append(r["output_tokens"])
        ratio = mean(min(v) / mean(v) for v in per_question.values()) if per_question else 0
        if positions:
            print(f"{source:15s} {len(correct):5d} {len(positions) / len(correct):6.0%} "
                  f"{median(positions):7.0%} {median(tail_tokens):7.0f} "
                  f"{median(checks_after):7.0f} {ratio:14.0%}")
    print("\n  found  = share of correct traces whose thinking mentions the gold answer")
    print("  first@ = median position of that first mention, as a share of the thinking")
    print("  after  = median thinking tokens after it;  checks = median Wait/check restarts after it")
    print("  shortest/mean = shortest correct sample of a question vs its mean correct sample")


if __name__ == "__main__":
    main(sys.argv[1])
