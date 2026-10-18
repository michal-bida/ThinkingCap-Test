"""Label-free look at when Qwen3-0.6B could stop thinking, and at token-costly wording.

    python analyze_cues.py samples/train3k.jsonl

No gold answer is used to find anything in the thinking. Gold is only used afterwards,
to score what was found.

1. Answer revisions. Every explicit answer statement in the thinking ("the answer is 12",
   "total is 12", "Answer: B", ...) is a candidate. Does the first candidate ever change,
   and when it does, does the change fix or break the answer? If the first candidate is
   as accurate as the final answer, everything after it is confirmation, and "stop at your
   first stated answer" is a cue the model could learn without knowing the gold.

2. Word cost. Which words of the thinking take more than one token, and how many tokens
   would go if each were replaced by a one-token word: the ceiling of a synonym rewrite.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from statistics import mean, median

from transformers import AutoTokenizer

from analyze_traces import lenient_correct, thinking_of, BOXED, ANSWER_IS
from utils.train.rewards import clean_answer, same_choice, same_number

NUMBER = r"\$?\s*(-?\d[\d,]*(?:\.\d+)?)"
# Only explicit answer statements: "total is 30" is as often an intermediate sum.
MATH_STATEMENT = re.compile(
    r"(?i:\banswer\b)[^.\n]{0,30}?(?:\bis\b|\bshould be\b|\bwould be\b|=|:)\s*\**" + NUMBER)
# The prompt's own format example ("like: Final answer: 42") is not a candidate.
FORMAT_EXAMPLE = re.compile(r"(?i)\blike\b|e\.g\.|for example|format")
CHOICE_STATEMENT = re.compile(
    r"(?i:answer|option|choice)\s*(?:is|would be|should be|must be|:)\s*\**\(?([A-E])\b")
WORD = re.compile(r"[A-Za-z']+")


HEDGE = re.compile(r"(?i)\b(?:maybe|perhaps|might|could|possibly|or)\b|\?")


def committed(thinking, end):
    """The answer statement ending at end is not hedged: no maybe/could/or/? in its sentence."""
    start = max(thinking.rfind(".", 0, end), thinking.rfind("\n", 0, end)) + 1
    stop = re.compile(r"[.!?\n]").search(thinking, end)
    sentence = thinking[start:stop.end() if stop else len(thinking)]
    return not HEDGE.search(sentence)


def candidates(record, only_committed=False):
    """(char_end, value) of each answer statement in the thinking, in order."""
    thinking = thinking_of(record["text"])
    pattern = MATH_STATEMENT if record["task"] == "math" else CHOICE_STATEMENT
    found = []
    for m in pattern.finditer(thinking):
        line_start = thinking.rfind("\n", 0, m.start()) + 1
        if FORMAT_EXAMPLE.search(thinking[max(line_start, m.start() - 60):m.end()]):
            continue
        value = m.group(1).replace(",", "")
        # "the final answer should be 42?": the prompt's example, echoed in confusion.
        if record["task"] == "math" and value == "42":
            continue
        if only_committed and not committed(thinking, m.end()):
            continue
        found.append((m.end(), value))
    return found


def same(record, a, b):
    if record["task"] == "math":
        return same_number(clean_answer(a), b)
    return same_choice(clean_answer(a), b)


def final_answer(record):
    """The model's own answer from its reply after </think>, or None."""
    if "</think>" not in record["text"]:
        return None
    reply = record["text"].split("</think>")[-1].replace("**", "")
    # "Final Answer:" can stand alone with a \boxed{} below it: take the first kind that
    # actually holds a value.
    found = None
    for kind in (re.findall(r"(?i)final answer:[ \t]*([^\n]+)", reply), BOXED.findall(reply),
                 ANSWER_IS.findall(reply)):
        kind = [k for k in kind if re.search(r"[A-Za-z0-9]", k)]
        if kind:
            found = kind
            break
    if not found:
        return None
    answer = re.sub(r"(?i)^(?:final answer:\s*)+", "", clean_answer(found[-1].strip()))
    if record["task"] == "math":
        numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", answer)
        return numbers[-1].replace(",", "") if numbers else answer
    return answer


def revisions(records, tokenizer, only_committed):
    kind = "first COMMITTED statement (no maybe/could/or/?)" if only_committed else "first statement"
    print(f"1. Answer revisions inside the thinking: {kind}\n")
    print(f"{'source':15s} {'n':>5s} {'has cand':>9s} {'1st right':>10s} {'final right':>12s} "
          f"{'changed':>8s} {'fixed':>6s} {'broke':>6s} {'tok after 1st':>14s} {'of thinking':>12s}")
    for source in sorted({r["source"] for r in records}):
        rows = [r for r in records if r["source"] == source and final_answer(r) is not None]
        found = [(r, c) for r in rows if (c := candidates(r, only_committed))]
        # Both scored on what the model said, not on the grader's format check.
        first_right = [same(r, c[0][1], r["answer"]) for r, c in found]
        final_right = [same(r, final_answer(r), r["answer"]) for r, _ in found]
        changed = [not same(r, c[0][1], final_answer(r)) for r, c in found]
        fixed = sum(not f and g for f, g in zip(first_right, final_right))
        broke = sum(f and not g for f, g in zip(first_right, final_right))
        after, share = [], []
        for r, c in found:
            thinking = thinking_of(r["text"])
            total = len(tokenizer(thinking, add_special_tokens=False)["input_ids"])
            tail = len(tokenizer(thinking[c[0][0]:], add_special_tokens=False)["input_ids"])
            after.append(tail)
            share.append(tail / max(1, total))
        print(f"{source:15s} {len(rows):5d} {len(found) / len(rows):9.0%} "
              f"{mean(first_right):10.1%} {mean(final_right):12.1%} {mean(changed):8.1%} "
              f"{fixed / len(found):6.1%} {broke / len(found):6.1%} {median(after):14.0f} "
              f"{mean(share):12.0%}")
    print("\n  has cand = share with such a statement in the thinking")
    print("  1st right / final right = the statement's answer vs the reply's answer, both vs gold")
    print("  fixed / broke = first wrong -> final right, first right -> final wrong")
    print("  tok after 1st = median thinking tokens after it; of thinking = mean share they make up\n")


def word_cost(records, tokenizer):
    print("2. Word cost of the thinking\n")
    words = Counter()
    for r in records:
        words.update(w.lower() for w in WORD.findall(thinking_of(r["text"])))
    cost = {w: len(tokenizer(" " + w, add_special_tokens=False)["input_ids"]) for w in words}
    total_words = sum(words.values())
    total_tokens = sum(n * cost[w] for w, n in words.items())
    extra = sum(n * (cost[w] - 1) for w, n in words.items())
    print(f"  {total_words} words, {total_tokens} word tokens, {total_tokens / total_words:.3f} tokens/word")
    print(f"  ceiling of a synonym rewrite (every multi-token word -> 1 token): "
          f"{extra / total_tokens:.1%} of word tokens\n")
    print("  costliest multi-token words (extra tokens they spend):")
    for w, n in sorted(((w, n) for w, n in words.items() if cost[w] > 1),
                       key=lambda item: -item[1] * (cost[item[0]] - 1))[:25]:
        print(f"    {w:20s} x{n:6d}  {cost[w]} tokens")
    print()


def main(path):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    records = [json.loads(line) for line in open(path)]
    revisions(records, tokenizer, only_committed=False)
    revisions(records, tokenizer, only_committed=True)
    word_cost(records, tokenizer)


if __name__ == "__main__":
    main(sys.argv[1])
