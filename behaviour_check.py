"""Does the base 0.6B behave on the training prompts as it does on the benchmarks?

    python behaviour_check.py sample --per-domain 300 --out samples/behaviour.jsonl   # GPU
    python behaviour_check.py sample --per-domain 0 --samples 4 --out samples/domains.jsonl
    python behaviour_check.py grade samples/behaviour.jsonl --base runs/base-s2      # CPU

Prompt length is only a proxy for matching the evaluation. This samples the base model once
per prompt on a fixed subset of data/domains, with the evaluation's sampling settings and token
caps, grades each reply with the evaluation's own graders (KodCode's tests for code), and puts
accuracy, output and thinking length and truncation beside the base model's run on the
benchmarks. A domain whose training prompts make the model think much longer or shorter, or
fail much more often, than the benchmark does is still mismatched.
"""

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from utils.eval import sandbox
from utils.eval.benchmarks import paths
from utils.eval.graders import (grade_call, grade_numeric, load_bfcl, load_ifeval, load_sanitize,
                                mbpp_code, split_thinking)
from utils.eval.recipe import SAMPLING, SEED, token_cap

MODEL = "Qwen/Qwen3-0.6B"
DOMAINS = ("code", "tools", "ifeval", "math")
BENCHMARK = {"code": "mbpp", "ifeval": "ifeval", "math": "svamp"}
TOOL_BENCHMARK = {"simple": "bfcl_simple_python", "multiple": "bfcl_multiple",
                  "irrelevance": "bfcl_irrelevance", "decline": None}
WORKER = Path(__file__).resolve().with_name("kodcode_worker.py")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample")
    sample.add_argument("--data", default="data/domains")
    sample.add_argument("--per-domain", type=int, default=300, help="0 = every row")
    sample.add_argument("--samples", type=int, default=1, help="Samples per prompt")
    sample.add_argument("--out", required=True)
    sample.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    grade = commands.add_parser("grade")
    grade.add_argument("samples")
    grade.add_argument("--data", default="data/domains")
    grade.add_argument("--base", default="runs/base-s2")
    grade.add_argument("--report", default="reports/behaviour_check.md")
    return parser.parse_args()


def subset(data, per_domain):
    """{domain: [row, ...]}: the first per_domain rows of a fixed shuffle of each domain."""
    chosen = {}
    for domain in DOMAINS:
        rows = [json.loads(line) for line in open(Path(data) / f"{domain}.jsonl")]
        random.Random(f"behaviour-{domain}").shuffle(rows)
        chosen[domain] = rows[:per_domain] if per_domain else rows
    return chosen


def benchmark_of(row):
    if row["task"] == "tools":
        return TOOL_BENCHMARK[row["meta"]["category"]]
    return BENCHMARK[row["task"]]


def cap(row):
    return token_cap(BENCHMARK.get(row["task"], "bfcl"))


# ---------------------------------------------------------------- sample (GPU)

def sample(arguments):
    import os
    import shutil
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # as evaluate.py: no nvcc here
    from vllm import LLM, SamplingParams

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = ({(r["task"], r["qid"], r.get("sample", 0)) for r in map(json.loads, out.open())}
            if out.exists() else set())
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    jobs = {}  # sample index outermost: vLLM runs requests first come, first served, so
    # every prompt gets its first samples before any gets its last
    chosen = subset(arguments.data, arguments.per_domain)
    prompts = {(d, row["qid"]): tokenizer.apply_chat_template(row["prompt"], tokenize=True,
                                                              add_generation_prompt=True, enable_thinking=True)
               for d, rows in chosen.items() for row in rows}
    for sample_index in range(arguments.samples):
        for domain, rows in chosen.items():
            for row in rows:
                if (domain, row["qid"], sample_index) not in done:
                    jobs[f"{domain}-{row['qid']}-{sample_index}"] = (row, prompts[(domain, row["qid"])], sample_index)
    print(f"{len(done)} already in {out}, {len(jobs)} to go", flush=True)
    if not jobs:
        return
    longest = max(len(ids) + cap(row) for row, ids, _ in jobs.values())
    engine = LLM(model=MODEL, max_model_len=longest,
                 gpu_memory_utilization=arguments.gpu_memory_utilization,
                 generation_config="vllm", seed=SEED, disable_log_stats=True,
                 attention_backend=None if shutil.which("nvcc") else "TRITON_ATTN").llm_engine
    for request_id, (row, ids, sample_index) in jobs.items():
        seed = SEED + DOMAINS.index(row["task"]) * 1_000_000 + row["qid"] * 1000 + sample_index
        engine.add_request(request_id, {"prompt_token_ids": ids},
                           SamplingParams(max_tokens=cap(row), seed=seed, **SAMPLING))
    eos_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    open_id, close_id = (tokenizer.convert_tokens_to_ids(t) for t in ("<think>", "</think>"))
    finished, generated, start = 0, 0, time.time()
    with out.open("a") as stream:
        while engine.has_unfinished_requests():
            for output in engine.step():
                if not output.finished:
                    continue
                row, ids, sample_index = jobs[output.request_id]
                tokens = list(output.outputs[0].token_ids)
                final, reasoning, boundary = split_thinking(tokens, open_id, close_id, eos_ids)
                stream.write(json.dumps({
                    "task": row["task"], "qid": row["qid"], "sample": sample_index, "source": row["source"],
                    "benchmark": benchmark_of(row), "prompt_tokens": len(ids),
                    "output_tokens": len(tokens), "reasoning_tokens": reasoning,
                    "truncated": not (tokens and tokens[-1] in eos_ids), "boundary_error": boundary,
                    "text": tokenizer.decode(tokens, skip_special_tokens=True),
                    "response": tokenizer.decode(final, skip_special_tokens=True),
                    "token_ids": tokens}) + "\n")
                stream.flush()
                finished += 1
                generated += len(tokens)
                if finished % 50 == 0 or finished == len(jobs):
                    elapsed = time.time() - start
                    print(f"{finished}/{len(jobs)}  {generated / elapsed:.0f} tok/s  "
                          f"{elapsed / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- grade (CPU)

def worker_command():
    """bubblewrap around kodcode_worker.py: no network, read-only, 4 GB, 120 s CPU per batch."""
    environment, runtime = Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()
    command = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for path in dict.fromkeys([Path("/usr"), Path("/lib"), Path("/lib64"), environment, runtime, WORKER]):
        if path.exists():
            command += ["--ro-bind", str(path), str(path)]
    command += sandbox.runtime_alias(environment, runtime)
    command += ["--chdir", "/tmp", "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin:/bin",
                "--setenv", "PYTHONHASHSEED", "0",
                "/usr/bin/prlimit", f"--as={sandbox.MEMORY_BYTES}", f"--cpu={sandbox.CPU_SECONDS}", "--",
                str(environment / "bin/python"), str(WORKER)]
    return command


def run_tests(items, batch=25):
    """[{"code", "test"}] -> verdicts, in batches so one runaway item costs one batch."""
    if not sandbox.have_bwrap():
        raise RuntimeError("bubblewrap is needed to run model-written code")
    verdicts = []
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        payload = "".join(json.dumps(item) + "\n" for item in chunk)
        try:
            done = subprocess.run(worker_command(), input=payload, text=True,
                                  capture_output=True, timeout=30 * len(chunk))
            got = [json.loads(line)["verdict"] for line in done.stdout.splitlines() if line.strip()]
        except subprocess.TimeoutExpired:
            got = []
        verdicts += got + ["crash"] * (len(chunk) - len(got))
    return verdicts


def grade(arguments):
    root = paths("questions")
    records = [json.loads(line) for line in open(arguments.samples)]
    rows = {(d, json.loads(line)["qid"]): json.loads(line)
            for d in DOMAINS for line in open(Path(arguments.data) / f"{d}.jsonl")}
    sanitize = load_sanitize(root["evalplus"])
    checker = load_bfcl(root["bfcl"])
    ifeval = load_ifeval(root["ifeval"])

    programs = []
    for record in records:
        row = rows[(record["task"], record["qid"])]
        record["correct"] = None
        try:
            if record["task"] == "math":
                record["correct"] = grade_numeric(record["text"], row["answer"], record["truncated"])[0]
            elif record["task"] == "tools":
                category = row["meta"]["category"]
                bfcl_row = {"function": row["meta"]["function"], "reference": row["meta"]["reference"],
                            "category": {"simple": "simple_python", "decline": "irrelevance"}.get(category, category)}
                record["correct"] = grade_call(bfcl_row, record["response"], *checker,
                                               truncated=record["truncated"])[0]
            elif record["task"] == "ifeval":
                ifeval_row = {"id": f"train-{row['qid']}", "original_prompt": row["prompt"][-1]["content"],
                              "instruction_id_list": row["meta"]["instruction_id_list"],
                              "kwargs": row["meta"]["kwargs"]}
                record["correct"] = ifeval(ifeval_row, record["response"])[0] and not record["truncated"]
            else:
                code, error = mbpp_code(record["response"], row["meta"]["entry_point"], sanitize,
                                        record["truncated"])
                if error:
                    record["correct"] = False
                else:
                    programs.append((record, {"code": code, "test": row["meta"]["test"]}))
        except Exception as error:  # a grader that cannot read a training row leaves it ungraded
            record["grading_error"] = f"{type(error).__name__}: {error}"[:200]
    print(f"running {len(programs)} code replies against KodCode's tests", flush=True)
    for (record, _), verdict in zip(programs, run_tests([item for _, item in programs])):
        record["verdict"] = verdict
        record["correct"] = None if verdict == "unsupported" else verdict == "pass"

    base = [json.loads(line) for line in open(Path(arguments.base) / "records.jsonl")]
    write_report(records, base, arguments)
    graded = Path(arguments.samples).with_suffix(".graded.jsonl")
    with graded.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    print(f"report -> {arguments.report}, graded records -> {graded}")


def summary(records):
    graded = [r for r in records if r.get("correct") is not None]
    tokens = [r["output_tokens"] for r in records]
    thinking = [r["reasoning_tokens"] for r in records if r["reasoning_tokens"] is not None]
    return {"n": len(records), "graded": len(graded),
            "accuracy": 100 * sum(bool(r["correct"]) for r in graded) / len(graded) if graded else float("nan"),
            "mean": statistics.mean(tokens), "median": statistics.median(tokens),
            "thinking": statistics.mean(thinking) if thinking else float("nan"),
            "truncated": 100 * sum(bool(r["truncated"]) for r in records) / len(records)}


def line(name, s):
    return (f"| {name} | {s['n']} ({s['graded']} graded) | {s['accuracy']:.1f} | {s['mean']:.0f} | "
            f"{s['median']:.0f} | {s['thinking']:.0f} | {s['truncated']:.1f} |")


def write_report(records, base, arguments):
    lines = ["# Behaviour check: base Qwen3-0.6B on the training prompts vs on the benchmarks", "",
             f"Training side: `{arguments.samples}` ({len({(r['task'], r['qid']) for r in records})} prompts, "
             f"{len(records)} samples). Benchmark side: "
             f"`{arguments.base}` (same model, same sampling and caps, 2 samples per prompt).", "",
             "| set | n | accuracy % | mean tokens | median tokens | mean thinking | truncated % |",
             "|---|---|---:|---:|---:|---:|---:|"]
    groups = defaultdict(list)
    for record in records:
        groups[(record["task"], record["benchmark"])].append(record)
        if record["task"] == "tools":
            groups[("tools", "all")].append(record)
    base_groups = defaultdict(list)
    for record in base:
        base_groups[record["dataset"]].append(record)
        if record["dataset"].startswith("bfcl"):
            base_groups["all"].append(record)
    for (task, benchmark), group in sorted(groups.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        lines.append(line(f"**train {task}**" + (f" ({benchmark})" if task == "tools" else ""), summary(group)))
        if benchmark and benchmark in base_groups:
            lines.append(line(f"eval {benchmark if benchmark != 'all' else 'bfcl (all)'}",
                              summary(base_groups[benchmark])))
    by_source = defaultdict(list)
    for record in records:
        by_source[record["source"]].append(record)
    lines += ["", "Per source (training side only):", "",
              "| source | n | accuracy % | mean tokens | median tokens | mean thinking | truncated % |",
              "|---|---|---:|---:|---:|---:|---:|"]
    lines += [line(source, summary(group)) for source, group in sorted(by_source.items())]
    verdicts = defaultdict(int)
    for record in records:
        if "verdict" in record:
            verdicts[record["verdict"]] += 1
    errors = sum("grading_error" in r for r in records)
    lines += ["", f"Code verdicts: {dict(verdicts)}. Rows a grader could not read: {errors}.", ""]
    Path(arguments.report).write_text("\n".join(lines) + "\n")


def main():
    arguments = parse_arguments()
    sample(arguments) if arguments.command == "sample" else grade(arguments)


if __name__ == "__main__":
    main()
