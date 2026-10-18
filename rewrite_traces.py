"""Arm R: Claude edits Qwen3-0.6B's own thinking by deletion and condensation only.

    python rewrite_traces.py data/arms/A.jsonl --out data/arms/R_raw.jsonl [--limit 20]

The edit rules come from comparing ThinkingCap-Qwen3.8-27B with its base model
(reports/FINDINGS.md, section 4) and from the 0.6B's own stopping cue (section 2): keep
the steps that reach the answer, end at the first committed answer, drop everything that
only repeats, hedges, wanders or talks about the output format. Nothing may be added, so
the result stays reasoning the 0.6B produced itself.

Reads the ANTHROPIC_API_KEY or CLAUDE_API_KEY environment variable. Resumable: qids
already in --out are skipped. Prints the token usage and an estimated cost.
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

from analyze_traces import thinking_of

MODEL = "claude-sonnet-5-5"
PRICE = {"input": 3.0, "output": 15.0}  # USD per million tokens, Sonnet-class estimate

SYSTEM = """You edit the reasoning ("thinking") a small language model wrote before answering a question. The edited thinking will be used to teach the same small model to think shorter, so it must stay reasoning that model actually produced.

Edit by DELETION and CONDENSATION only:
- Keep the model's own steps, facts and arithmetic that lead to its answer. Never add a fact, step, check or argument that is not in the original.
- If the original made an error and later corrected it, keep only the corrected path.

Remove:
- restating or copying the question or the answer options;
- any talk about the output format (the "Final answer:" line, the example in the prompt, what the user wants the reply to look like);
- filler and meta talk ("Okay", "Let me think", "Hmm", "I need to figure out", "Let me make sure");
- detours into readings or options that the model later drops, except one short clause on why the closest rejected option fails;
- repeated computations and re-checks of results already established;
- everything after the first point where the model commits to its final answer. On a multi-step math problem, keep at most one brief check, and only if the original contains one.

Style: terse and telegraphic. Drop articles and filler words where meaning stays clear, keep numbers and equations, plain text (no Markdown, no LaTeX). The edited thinking must end with a committed statement of the same answer the original reached, e.g. "So answer 53." or "Answer: B."

Return only the edited thinking inside <edited></edited> tags."""


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arm", help="Arm file whose thinking is edited, e.g. data/arms/A.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, help="Edit only the first N rows (spread over sources)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default=MODEL)
    return parser.parse_args()


def spread(rows, limit):
    """The first `limit` rows, taken round-robin over sources."""
    by_source = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    out = []
    while len(out) < limit and any(by_source.values()):
        for source in sorted(by_source):
            if by_source[source] and len(out) < limit:
                out.append(by_source[source].pop(0))
    return out


def edit(client, model, row):
    completion = row["completion"][0]["content"]
    thinking = thinking_of(completion).strip()
    answer = completion.split("</think>", 1)[1].strip()
    final = re.findall(r"Final answer:\s*([^\n]+)", answer)
    message = (f"QUESTION:\n{row['prompt'][0]['content']}\n\n"
               f"THE MODEL'S FINAL ANSWER: {final[-1].strip() if final else row['answer']}\n\n"
               f"ORIGINAL THINKING:\n{thinking}")
    for attempt in range(5):
        try:
            response = client.messages.create(model=model, max_tokens=4096, system=SYSTEM,
                                              messages=[{"role": "user", "content": message}])
            break
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as error:
            if attempt == 4:
                raise
            time.sleep(10 * (attempt + 1))
            print(f"retry {row['qid']}: {type(error).__name__}", file=sys.stderr, flush=True)
    text = response.content[0].text
    match = re.search(r"<edited>(.*?)</edited>", text, re.DOTALL)
    return {"qid": row["qid"], "source": row["source"], "original": thinking,
            "edited": match.group(1).strip() if match else None, "answer": answer,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens}


def main():
    arguments = parse_arguments()
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
    client = anthropic.Anthropic(api_key=key)
    rows = [json.loads(line) for line in open(arguments.arm)]
    if arguments.limit:
        rows = spread(rows, arguments.limit)
    out = Path(arguments.out)
    done = {json.loads(line)["qid"] for line in out.open()} if out.exists() else set()
    rows = [row for row in rows if row["qid"] not in done]
    print(f"{len(done)} already edited, {len(rows)} to go", flush=True)

    used = {"input": 0, "output": 0}
    with out.open("a") as stream, ThreadPoolExecutor(arguments.workers) as pool:
        futures = [pool.submit(edit, client, arguments.model, row) for row in rows]
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            used["input"] += result["input_tokens"]
            used["output"] += result["output_tokens"]
            if count % 50 == 0 or count == len(rows):
                cost = sum(used[k] * PRICE[k] / 1e6 for k in used)
                print(f"{count}/{len(rows)}  tokens in {used['input']} out {used['output']}  "
                      f"~${cost:.2f}", flush=True)


if __name__ == "__main__":
    main()
