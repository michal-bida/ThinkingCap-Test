"""Sample traces from a model served by llama.cpp's llama-server, in sample_traces.py's format.

    llama-server -m model.gguf --parallel 4 ...        # separately, see the README of this run
    python sample_llama.py --model-name thinkingcap --qids samples/pilot.jsonl \
        --sources gsm8k,arc_easy,commonsense_qa --out samples/thinkingcap.jsonl

The questions are those of an earlier sample file (--qids), so traces of different
models can be compared question by question. Sampling is the evaluation's (temperature
0.6, top-p 0.95, top-k 20) unless --temperature says otherwise. Rows already in --out are skipped, so an
interrupted run resumes.
"""

import argparse
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from utils.eval.recipe import SAMPLING, SEED
from utils.train.rewards import is_task_correct


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", required=True, help="Label stored in every row")
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--qids", required=True, help="Sample file whose questions to reuse")
    parser.add_argument("--sources", help="Comma-separated sources to keep (default: all)")
    parser.add_argument("--limit", type=int, help="At most this many questions")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=SAMPLING["temperature"],
                        help="Qwen3.8 recommends 1.0 for thinking; Qwen3 0.6 (the default)")
    parser.add_argument("--reasoning-effort", choices=["xhigh", "medium", "low"],
                        help="Qwen3.8 chat-template effort level (default: the template's, xhigh)")
    parser.add_argument("--parallel", type=int, default=4, help="Match llama-server --parallel")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def questions(arguments):
    sources = set(arguments.sources.split(",")) if arguments.sources else None
    wanted = []
    for row in map(json.loads, open(arguments.qids)):
        if row["sample"] == 0 and (sources is None or row["source"] in sources):
            wanted.append(row["qid"])
    # Interleave sources so a partial run still covers all of them.
    rows = {qid: json.loads(line) for qid, line in enumerate(open(arguments.data))}
    by_source = {}
    for qid in wanted:
        by_source.setdefault(rows[qid]["source"], []).append(qid)
    order = [qid for group in zip(*by_source.values()) for qid in group]
    order += [qid for qid in wanted if qid not in set(order)]
    return [(qid, rows[qid]) for qid in order[:arguments.limit]]


def complete(server, row, seed, max_tokens, temperature, effort):
    body = {"messages": row["prompt"], "max_tokens": max_tokens, "seed": seed,
            "temperature": temperature, "top_p": SAMPLING["top_p"],
            "top_k": SAMPLING["top_k"], "min_p": 0.0,
            # Keep <think>...</think> in the content, as vLLM returns it for Qwen3-0.6B.
            "reasoning_format": "none", "chat_template_kwargs": {"enable_thinking": True}}
    if effort:
        body["chat_template_kwargs"]["reasoning_effort"] = effort
    request = urllib.request.Request(f"{server}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=6 * 3600) as response:
        return json.load(response)


def main():
    arguments = parse_arguments()
    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["qid"], r["sample"]) for r in map(json.loads, out.open())}
    jobs = [(qid, row, sample) for qid, row in questions(arguments)
            for sample in range(arguments.samples) if (qid, sample) not in done]
    print(f"{len(done)} rows already in {out}, {len(jobs)} to go", flush=True)

    lock, finished, generated, start = threading.Lock(), 0, 0, time.time()

    def run(job):
        qid, row, sample = job
        seed = SEED + qid * 1000 + sample
        reply = complete(arguments.server, row, seed, arguments.max_tokens,
                         arguments.temperature, arguments.reasoning_effort)
        choice, usage = reply["choices"][0], reply["usage"]
        text = choice["message"]["content"]
        thinking = text.split("</think>")[0] if "</think>" in text else None
        return {"model": arguments.model_name, "qid": qid, "source": row["source"],
                "task": row["task"], "answer": row["answer"], "sample": sample, "seed": seed,
                "prompt_tokens": usage["prompt_tokens"], "text": text,
                "output_tokens": usage["completion_tokens"],
                "thinking_chars": len(thinking) if thinking is not None else None,
                "truncated": choice["finish_reason"] == "length",
                "correct": is_task_correct(text, row["answer"], row["task"]),
                "seconds": reply.get("timings", {}).get("predicted_ms", 0) / 1000}

    with out.open("a") as stream, ThreadPoolExecutor(arguments.parallel) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            record = future.result()
            with lock:
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                finished += 1
                generated += record["output_tokens"]
                elapsed = time.time() - start
                print(f"{finished}/{len(jobs)}  {record['source']:15s} {record['output_tokens']:5d} tok  "
                      f"correct={record['correct']!s:5s}  total {generated / elapsed:.1f} tok/s  "
                      f"{elapsed / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
