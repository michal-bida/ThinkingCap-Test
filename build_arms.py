"""Build the SFT training sets of the experiment's arms from sampled traces.

    python build_arms.py samples/train3k.jsonl --out data/arms

Only strictly correct, finished traces are used (the reply ends with the required
"Final answer:" line), so no arm teaches a broken answer format. Every arm covers the
same questions: those with at least one such trace.

    A0  a random correct trace          (control: fine-tuning on own samples at all)
    A   the shortest correct trace      (selection only)
    B   A with the thinking cut after the sentence where the gold answer first appears
        (diagnostic only: the cut is chosen with the gold answer)
    S   a correct trace cut after the sentence of its first COMMITTED answer (an unhedged
        answer statement, or for math its own final answer stated as a result; no gold). Only traces whose first committed
        answer is their final answer qualify; the shortest after the cut is used, else A.
    T   S with ThinkingCap-style removals done by rules (compress()): no format talk, no
        filler or re-check announcements, no copied question, no articles.
    U   T with the reply after </think> reduced to the bare "Final answer: X" line.

The answer after </think> is kept verbatim. Rows are in TRL's conversational
prompt-completion format: {"prompt": [user], "completion": [assistant], ...}.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer

from analyze_cues import candidates, committed, final_answer, same
from analyze_traces import first_gold_end, gold_pattern, thinking_of


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def committed_cut(trace, question):
    """Thinking up to the end of the sentence holding the first committed answer, or None.

    A commitment is an unhedged explicit answer statement ("so the answer is 48"), or, for
    math, an unhedged sentence where the trace's OWN final answer appears as a result
    ("Jake needs to buy 48 papayas"). The gold answer is not used.
    """
    final = final_answer(trace)
    if final is None:
        return None
    thinking = thinking_of(trace["text"])
    found = candidates(trace, only_committed=True)
    if trace["task"] == "math":
        own = gold_pattern(dict(trace, answer=final), question)
        for m in own.finditer(thinking):
            rest = re.match(r"[^.!?\n]*", thinking[m.end():]).group(0)
            # The answer must be the sentence's result: no further number after it.
            if committed(thinking, m.end()) and not re.search(r"\d", rest):
                found.append((m.end(), final))
    if not found:
        return None
    end, value = min(found)
    if not same(trace, value, final):
        return None
    boundary = re.compile(r"[.!?](?:\s|$)|\n").search(thinking, end)
    cut = thinking[:boundary.end() if boundary else len(thinking)]
    # Don't end the thinking on the prompt's format example ("like: Final answer: 42").
    last_sentence = cut[max(cut.rfind(". ", 0, end), cut.rfind("\n", 0, end)) + 1:]
    if re.search(r"\b42\b|\bexample\b", last_sentence) and final != "42":
        return None
    return cut


FORMAT_TALK = re.compile(r"(?i)final answer|last line|\bformat\b|\bexample\b|\b42\b|"
                         r"\bthe user (?:wants|says|said|asks|asked)")
FILLER_SENTENCE = re.compile(
    r"(?i)^(?:okay|ok|alright|hmm+|well|so|wait|right|yes|yeah|yep|got it|good|great|"
    r"let me (?:think|see|check|verify|confirm|make sure|double-?check|re-?check|start|try)"
    r"(?: (?:again|this|that|it|once more|about (?:this|that|it)))?|"
    r"let's (?:see|think|check|start)|i (?:think|need to figure (?:this|it) out)|"
    r"that (?:makes sense|seems right|seems correct|checks out))[\s,.!:]*$")
LEADING_FILLER = re.compile(r"^(?:(?:okay|ok|alright|hmm+|well|so|now|wait|right|"
                            r"let me see|let's see)\b[,.!]?\s*)+", re.IGNORECASE)
# Lowercase articles, and "The"/"An" starting a sentence; never "A" (a variable or option).
ARTICLE = re.compile(r"\b(?:the|a|an|The|An) (?=[a-z])")
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


def ngram_set(text, n=4):
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def compress(thinking, question, articles=False):
    """ThinkingCap-style removals by rule, sentence by sentence; nothing is added.

    articles=True keeps "the/a/an" (arm V): without them the 0.6B loops (arm T).
    """
    question_grams = ngram_set(question)
    kept = []
    for sentence in SENTENCE_BREAK.split(thinking):
        text = sentence.strip()
        if not text:
            continue
        if FORMAT_TALK.search(text) or FILLER_SENTENCE.match(text):
            continue
        # Sentences that mostly copy the question, which stays in the context anyway.
        grams = ngram_set(text)
        if grams and len(grams & question_grams) / len(grams) > 0.5:
            continue
        text = LEADING_FILLER.sub("", text)
        if not articles:
            text = ARTICLE.sub("", text)
        if text:
            kept.append(text[0].upper() + text[1:])
    return " ".join(kept)


def completion(thinking, answer):
    return f"<think>\n{thinking.strip()}\n</think>\n\n{answer.strip()}"


def main():
    arguments = parse_arguments()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    rows = {qid: json.loads(line) for qid, line in enumerate(open(arguments.data))}
    by_question = defaultdict(list)
    for trace in map(json.loads, open(arguments.traces)):
        if trace["correct"] and not trace["truncated"] and "</think>" in trace["text"]:
            by_question[trace["qid"]].append(trace)

    rng = random.Random(arguments.seed)
    arms = defaultdict(list)
    for qid in sorted(by_question):
        traces = by_question[qid]
        row = rows[qid]
        question = row["prompt"][0]["content"]
        shortest = min(traces, key=lambda t: t["output_tokens"])
        cut = first_gold_end(shortest, question)
        thinking = thinking_of(shortest["text"])
        answer = shortest["text"].split("</think>", 1)[1]
        cuts = [(cut_text, t) for t in traces if (cut_text := committed_cut(t, question))]
        if cuts:
            s_thinking, s_trace = min(cuts, key=lambda pair: len(pair[0]))
        else:
            s_thinking, s_trace = thinking, shortest
        chosen = {
            "A0": completion(thinking_of((pick := rng.choice(traces))["text"]),
                             pick["text"].split("</think>", 1)[1]),
            "A": completion(thinking, answer),
            "B": completion(thinking[:cut] if cut else thinking, answer),
            "S": completion(s_thinking, s_trace["text"].split("</think>", 1)[1]),
            "T": completion(compress(s_thinking, question), s_trace["text"].split("</think>", 1)[1]),
            "U": completion(compress(s_thinking, question), f"Final answer: {final_answer(s_trace)}"),
            "V": completion(compress(s_thinking, question, articles=True),
                            s_trace["text"].split("</think>", 1)[1]),
        }
        for arm, text in chosen.items():
            arms[arm].append({"prompt": row["prompt"],
                              "completion": [{"role": "assistant", "content": text}],
                              "qid": qid, "source": row["source"], "answer": row["answer"],
                              "task": row["task"]})

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(by_question)} questions with a strictly correct finished trace")
    for arm, items in arms.items():
        with (out / f"{arm}.jsonl").open("w") as stream:
            for item in items:
                stream.write(json.dumps(item) + "\n")
        lengths = defaultdict(list)
        for item in items:
            text = item["completion"][0]["content"]
            think = len(tokenizer(thinking_of(text), add_special_tokens=False)["input_ids"])
            lengths[item["source"]].append(think)
        summary = "  ".join(f"{s} {mean(v):.0f}" for s, v in sorted(lengths.items()))
        print(f"{arm:3s} {len(items):5d} rows -> {out / arm}.jsonl   thinking tokens: {summary}")


if __name__ == "__main__":
    main()
