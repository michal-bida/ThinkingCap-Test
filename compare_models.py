"""Compare how different models think on the same questions.

    python compare_models.py --out reports/compare_models.md \
        qwen3-0.6b=samples/pilot.jsonl thinkingcap=samples/llama-thinkingcap.jsonl \
        qwen3.8=samples/llama-qwen3.8.jsonl qwen3.8-low=samples/llama-qwen3.8-low.jsonl

Only questions every model answered are compared (sample 0 of each), and every text is
re-tokenized with the Qwen3-0.6B tokenizer, so lengths are in the units that matter for
the model being trained. Writes a Markdown report: length and accuracy per source, style
measures of the thinking, and side-by-side examples.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from statistics import mean, median

from transformers import AutoTokenizer

from analyze_traces import gold_pattern, lenient_correct, thinking_of

FILLER = re.compile(r"\b(?:okay|alright|hmm|well|so|let me|let's|i think|maybe|wait|actually|"
                    r"i need to|i should|i'll)\b", re.IGNORECASE)
CHECK = re.compile(r"\b(?:wait|double-?check|verify|let me check|confirm|re-?check|once more|"
                   r"again)\b", re.IGNORECASE)
ARTICLES = re.compile(r"\b(?:the|a|an)\b", re.IGNORECASE)
FORMAT_TALK = re.compile(r"final answer|last line|format", re.IGNORECASE)
MARKUP = re.compile(r"\\\(|\\\[|\\frac|\\boxed|\$|\*\*|^#+ |^- ", re.MULTILINE)
SENTENCE = re.compile(r"[^.!?\n]+[.!?\n]")
WORD = re.compile(r"[A-Za-z']+")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="name=path, in the column order wanted")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--examples", type=int, default=2, help="Side-by-side examples per source")
    return parser.parse_args()


def ngrams(text, n=4):
    words = [w.lower() for w in WORD.findall(text)]
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def measure(record, question, tokenizer):
    text = record["text"]
    closed = "</think>" in text
    thinking = thinking_of(text).strip()
    answer = text.split("</think>", 1)[1].strip() if closed else ""
    words = WORD.findall(thinking)
    sentences = [s for s in SENTENCE.findall(thinking) if WORD.search(s)]
    think_tokens = len(tokenizer(thinking, add_special_tokens=False)["input_ids"])
    match = gold_pattern(record, question).search(thinking)
    question_grams = ngrams(question.split("\n\n")[0])
    # Restating: share of the opening 60 words that repeat 4-grams of the question.
    opening = " ".join(words[:60])
    restated = (len(ngrams(opening) & question_grams) / max(1, len(ngrams(opening)))
                if question_grams else 0.0)
    return {
        "correct": lenient_correct(record),
        "strict": record["correct"],
        "truncated": record["truncated"] or not closed,
        "think_tokens": think_tokens,
        "answer_tokens": len(tokenizer(answer, add_special_tokens=False)["input_ids"]),
        "words_per_sentence": len(words) / max(1, len(sentences)),
        "filler_per_100w": 100 * len(FILLER.findall(thinking)) / max(1, len(words)),
        "checks": len(CHECK.findall(thinking)),
        "articles_per_100w": 100 * len(ARTICLES.findall(thinking)) / max(1, len(words)),
        "format_talk": len(FORMAT_TALK.findall(thinking)),
        "markup": len(MARKUP.findall(thinking)),
        "restated": restated,
        "gold_at": (len(tokenizer(thinking[:match.end()], add_special_tokens=False)["input_ids"])
                    / max(1, think_tokens)) if match and thinking else None,
    }


def load(path):
    rows = {}
    for record in map(json.loads, open(path)):
        if record["sample"] == 0:
            rows[record["qid"]] = record
    return rows


def table(header, lines):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(cell) for cell in line) + " |" for line in lines]
    return "\n".join(out)


def main():
    arguments = parse_arguments()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    questions = {qid: json.loads(line)["prompt"][0]["content"]
                 for qid, line in enumerate(open(arguments.data))}
    runs = dict(run.split("=", 1) for run in arguments.runs)
    loaded = {name: load(path) for name, path in runs.items()}
    common = set.intersection(*(set(rows) for rows in loaded.values()))
    names = list(runs)

    metrics = {name: {qid: measure(loaded[name][qid], questions[qid], tokenizer) for qid in common}
               for name in names}
    sources = sorted({loaded[names[0]][qid]["source"] for qid in common})
    by_source = {s: [q for q in common if loaded[names[0]][q]["source"] == s] for s in sources}

    report = [f"# How the models think on the same {len(common)} questions",
              "", "Sample 0 of each model; lengths in Qwen3-0.6B tokens. Runs: " +
              ", ".join(f"`{n}` = {p}" for n, p in runs.items()), ""]

    report += ["## Length and accuracy", ""]
    lines = []
    for source in sources:
        qids = by_source[source]
        for name in names:
            m = [metrics[name][q] for q in qids]
            lines.append([source, name, len(m), f"{mean(x['correct'] for x in m):.0%}",
                          f"{mean(x['truncated'] for x in m):.0%}",
                          f"{mean(x['think_tokens'] for x in m):.0f}",
                          f"{median(x['think_tokens'] for x in m):.0f}",
                          f"{mean(x['answer_tokens'] for x in m):.0f}"])
    report.append(table(["source", "model", "n", "correct", "truncated", "thinking mean",
                         "thinking median", "answer mean"], lines))

    report += ["", "## Style of the thinking (means over finished traces)", "",
               "words/sentence; filler words (okay, hmm, so, let me, wait, maybe...) and articles "
               "per 100 words; re-check markers per trace; mentions of the answer format; "
               "Markdown/LaTeX marks; share of the opening that repeats the question; position "
               "where the gold answer is first committed (share of thinking).", ""]
    lines = []
    for source in sources:
        qids = by_source[source]
        for name in names:
            m = [metrics[name][q] for q in qids if not metrics[name][q]["truncated"]]
            if not m:
                continue
            gold = [x["gold_at"] for x in m if x["gold_at"] is not None and x["correct"]]
            lines.append([source, name, f"{mean(x['words_per_sentence'] for x in m):.1f}",
                          f"{mean(x['filler_per_100w'] for x in m):.1f}",
                          f"{mean(x['articles_per_100w'] for x in m):.1f}",
                          f"{mean(x['checks'] for x in m):.1f}",
                          f"{mean(x['format_talk'] for x in m):.1f}",
                          f"{mean(x['markup'] for x in m):.1f}",
                          f"{mean(x['restated'] for x in m):.0%}",
                          f"{median(gold):.0%}" if gold else "-"])
    report.append(table(["source", "model", "words/sent", "filler/100w", "articles/100w",
                         "checks", "format talk", "markup", "restated", "gold at"], lines))

    report += ["", "## Side by side", ""]
    rng = random.Random(0)
    for source in sources:
        for qid in rng.sample(sorted(by_source[source]), min(arguments.examples, len(by_source[source]))):
            report += [f"### {source}, question {qid} (gold: {loaded[names[0]][qid]['answer']})", "",
                       "```text", questions[qid][:1500], "```", ""]
            for name in names:
                m = metrics[name][qid]
                report += [f"**{name}**: {m['think_tokens']} thinking + {m['answer_tokens']} answer "
                           f"tokens, {'correct' if m['correct'] else 'wrong'}", "",
                           "```text", loaded[name][qid]["text"][:3000] +
                           ("\n[... cut for the report ...]" if len(loaded[name][qid]["text"]) > 3000 else ""),
                           "```", ""]

    with open(arguments.out, "w") as stream:
        stream.write("\n".join(report) + "\n")
    print(f"Wrote {arguments.out} ({len(common)} common questions)")


if __name__ == "__main__":
    main()
