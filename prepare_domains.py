"""Base training prompts for the four evaluated domains, in the evaluation's own prompt formats.

    python prepare_domains.py --out data/domains --per-domain 3000

Nothing here is an evaluation item. Each domain comes from a separate public dataset and is
written in the prompt format of the benchmark it stands in for:

  code   KodCode-V1 imperative tasks (with pytest tests)       -> MBPP+ format
  tools  ToolACE single-turn calls + xlam-irrelevance-7.5k    -> BFCL format (its own system prompt)
  ifeval IF_multi_constraints_upto5 + RLVR-IFeval (one constraint), IFEval ids -> IFEval format
  math   GSM8K train + MetaMathQA GSM_SV / GSM_FOBAR          -> SVAMP format ("Final answer:" line)

Every row keeps what a grader needs (tests, function list and reference call, instruction ids
and kwargs, numeric answer). A prompt sharing an 8-gram with any evaluation prompt is dropped;
for IFEval only the base request is compared, since the constraint sentences are IFEval's own
templates by design.

Prompt length is matched to the benchmark, not just capped: each source group is drawn
evenly from the deciles of its benchmark's prompt-token distribution (whole chat template,
so the shared instruction text cancels out), and rows outside the benchmark's length range
are dropped. Writes <out>/<domain>.jsonl and reports/domains_review.md.
"""

import argparse
import ast
import json
import random
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from utils.eval.graders import grade_call, load_bfcl
from utils.eval.benchmarks import MBPP_INSTRUCTION, MBPP_NAME_REMINDER, NUMERIC_INSTRUCTION, bfcl_system_prompt, paths

# The benchmark whose prompt lengths each source group is matched to. IFEval groups are
# matched to the IFEval prompts with the same number of instructions.
TARGET = {"code": "mbpp", "math": "svamp", "toolace/simple": "bfcl_simple_python",
          "toolace/decline": "bfcl_simple_python", "toolace/multiple": "bfcl_multiple",
          "xlam/irrelevance": "bfcl_irrelevance"}
LENGTH_BINS = 10


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/domains")
    parser.add_argument("--per-domain", type=int, default=3000)
    parser.add_argument("--review", default="reports/domains_review.md")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def grams(text, n=8):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


class Funnel:
    """Counts why rows were dropped, per domain, for the review."""

    def __init__(self):
        self.counts = defaultdict(Counter)

    def __call__(self, domain, reason):
        self.counts[domain][reason] += 1


# ---------------------------------------------------------------- code

def literal_assert(line, name):
    """True for `assert name(<literals>) == <literal>`: readable without the test's variables."""
    try:
        test = ast.parse(line).body[0].test
    except (SyntaxError, IndexError, AttributeError):
        return False
    if not (isinstance(test, ast.Compare) and isinstance(test.left, ast.Call)
            and call_name(test.left.func) == name):
        return False
    values = test.left.args + [k.value for k in test.left.keywords] + test.comparators
    try:
        for value in values:
            ast.literal_eval(value)
    except ValueError:
        return False
    return True


def first_assert(test, name):
    for line in test.splitlines():
        line = line.strip()
        if line.startswith("assert ") and len(line) < 200 and literal_assert(line, name):
            return line
    return None


# MBPP asks "Write a function to ...". KodCode's Prefill subset starts with a `def` stub
# (HumanEval style) and many Filter rows are chat questions ("How can I ...?", "Can you
# explain ..."); only tasks phrased as an instruction to write code are kept.
IMPERATIVE = re.compile(r"^(?:Please )?(?:Write|Create|Implement|Design|Develop|Define|Construct|"
                        r"Generate|Build|Devise|Given|You are given)\b")
CODE_SUBSETS = ("Filter", "Prefill", "Leetcode", "Evol")


def code_rows(funnel, rng):
    rows = []
    pool = load_dataset("KodCode/KodCode-V1", split="train")
    pool = pool.filter(lambda r: r["subset"] in CODE_SUBSETS, num_proc=8)
    for r in pool:
        question = r["question"].strip()
        if r["gpt_difficulty"] == "hard":
            funnel("code", "hard")
            continue
        if len(question) > 700:  # far above MBPP's longest task; spares tokenizing
            funnel("code", "question over 700 characters")
            continue
        if "```" in question or not IMPERATIVE.match(question):
            funnel("code", "not an instruction to write code (question, chat or def stub)")
            continue
        info = r["test_info"] or []
        info = ast.literal_eval(info) if isinstance(info, str) else info
        if len(info) != 1 or not info[0].get("function_name"):
            funnel("code", "not exactly one function")
            continue
        name = info[0]["function_name"]
        example = first_assert(r["test"], name)
        if not example:
            funnel("code", "no one-line assert on literals to show")
            continue
        task = f'"""\n{question}\n{example}\n"""\n'
        content = f"{MBPP_INSTRUCTION}\n\n{task}\n{MBPP_NAME_REMINDER.format(name=name)}"
        rows.append({"prompt": [{"role": "user", "content": content}], "task": "code",
                     "source": f"kodcode/{r['subset']}", "answer": "",
                     "meta": {"entry_point": name, "test": r["test"], "solution": r["solution"],
                              "difficulty": r["gpt_difficulty"], "id": r["question_id"]},
                     "_compare": question})
    return rows


# ---------------------------------------------------------------- tools

XLAM_TYPES = {"str": "string", "int": "integer", "float": "float", "bool": "boolean",
              "list": "array", "dict": "dict", "any": "any"}


def xlam_type(text):
    base = text.split(",")[0].strip().lower()
    for prefix, mapped in XLAM_TYPES.items():
        if base.startswith(prefix) or base.startswith(f"list[") and mapped == "array":
            return mapped
    if base.startswith("list") or base.startswith("tuple"):
        return "array"
    return "string"


def xlam_function(tool):
    """xLAM's flat parameter dict -> BFCL's JSON-schema-like function format."""
    properties, required = {}, []
    for name, spec in (tool.get("parameters") or {}).items():
        kind = spec.get("type", "str")
        prop = {"type": xlam_type(kind), "description": spec.get("description", "")}
        if "default" in spec and spec["default"] not in (None, ""):
            prop["default"] = spec["default"]
        elif "optional" not in kind.lower():
            required.append(name)
        properties[name] = prop
    return {"name": tool["name"], "description": tool.get("description", ""),
            "parameters": {"type": "dict", "properties": properties, "required": required}}


BFCL_TYPES = {"int": "integer", "str": "string", "bool": "boolean", "list": "array",
              "object": "dict", "number": "float", "tuple": "tuple", "any": "any"}


def bfcl_types(schema):
    """ToolACE writes "int", "str", "object", ...; BFCL's checker knows "integer", "string", "dict"."""
    if isinstance(schema, dict):
        if isinstance(schema.get("type"), str):
            schema["type"] = BFCL_TYPES.get(schema["type"], schema["type"])
        for value in schema.values():
            bfcl_types(value)
    elif isinstance(schema, list):
        for value in schema:
            bfcl_types(value)
    return schema


def toolace_functions(system):
    marker = "Here is a list of functions in JSON format that you can invoke:"
    if marker not in system:
        return None
    text = system.split(marker, 1)[1].strip()
    try:
        functions, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return None
    for function in functions:
        function.pop("required", None)  # a stray top-level key, always null
        bfcl_types(function.get("parameters"))
    return functions


UNSAFE = re.compile(r"[^A-Za-z0-9_.]")


def sanitize_names(functions, answer):
    """'Market Trends API' -> 'Market_Trends_API', in the function list and in the call."""
    for function in functions:
        clean = UNSAFE.sub("_", function["name"].strip())
        if clean != function["name"]:
            answer = answer.replace(function["name"] + "(", clean + "(")
            function["name"] = clean
    return functions, answer


IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = call_name(node.value)
        return inner and f"{inner}.{node.attr}"
    return None


def reference(call_text, functions):
    """'[f(a=1)]' -> BFCL possible-answer format, or None if it cannot be read."""
    try:
        tree = ast.parse(call_text.strip(), mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(tree, ast.List) or len(tree.elts) != 1 or not isinstance(tree.elts[0], ast.Call):
        return None
    call = tree.elts[0]
    name = call_name(call.func)
    by_name = {f["name"]: f for f in functions}
    if not name or name not in by_name or call.args:
        return None
    answer = {}
    try:
        for keyword in call.keywords:
            answer[keyword.arg] = [ast.literal_eval(keyword.value)]
    except ValueError:
        return None
    parameters = by_name[name]["parameters"]
    for param, spec in parameters.get("properties", {}).items():
        if param not in answer and param not in (parameters.get("required") or []):
            answer[param] = [""] + ([spec["default"]] if "default" in spec else [])
    return [{name: answer}]


def tools_rows(funnel, rng, root):
    rows = []
    for r in load_dataset("Team-ACE/ToolACE", split="train"):
        conversation = r["conversations"]
        if len(conversation) != 2:
            funnel("tools", "toolace multi-turn")
            continue
        functions = toolace_functions(r["system"])
        if not functions or len(functions) > 4:
            funnel("tools", "toolace: unreadable or >4 functions (BFCL has at most 4)")
            continue
        question, answer = conversation[0]["value"], conversation[1]["value"].strip()
        functions, answer = sanitize_names(functions, answer)
        if any(not IDENTIFIER.match(f["name"]) for f in functions):
            funnel("tools", "toolace: function name not a Python identifier")
            continue
        if answer.startswith("["):
            ref = reference(answer, functions)
            if ref is None:
                funnel("tools", "toolace: not exactly one readable call")
                continue
            category = "simple" if len(functions) == 1 else "multiple"
        else:
            ref, category = None, "decline"
        rows.append({"prompt": [{"role": "system", "content": bfcl_system_prompt(root, functions)},
                                {"role": "user", "content": question}],
                     "task": "tools", "source": f"toolace/{category}", "answer": answer,
                     "meta": {"category": category, "function": functions, "reference": ref},
                     "_compare": question})
    for r in load_dataset("MadeAgents/xlam-irrelevance-7.5k", split="train"):
        tools = json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"]
        if not tools or len(tools) > 2:
            funnel("tools", "xlam-irrelevance: no tools or >2 (BFCL irrelevance has 1)")
            continue
        functions = [xlam_function(t) for t in tools]
        if any(not IDENTIFIER.match(f["name"]) for f in functions):
            funnel("tools", "xlam-irrelevance: name not a Python identifier")
            continue
        rows.append({"prompt": [{"role": "system", "content": bfcl_system_prompt(root, functions)},
                                {"role": "user", "content": r["query"]}],
                     "task": "tools", "source": "xlam/irrelevance", "answer": "",
                     "meta": {"category": "irrelevance", "function": functions, "reference": None},
                     "_compare": r["query"]})
    kept = []
    for row in rows:
        if EXPLICIT.search(row["_compare"] + json.dumps(row["meta"]["function"])):
            funnel("tools", "explicit content")
        else:
            kept.append(row)
    return kept


# ---------------------------------------------------------------- instruction following

# IFEval asks for writing: essays, letters, stories, rewrites. These base sets do not.
IF_EXCLUDED_SOURCES = ("math", "jailbreak", "wildguard", "table_gpt", "science", "codealpaca", "code")


ENGLISH_WORDS = set("the a an and of to in is are be for on with that this it you your i what how "
                    "write about as by from or can do does please".split())


SEXUAL = re.compile(r"(?i)\b(in the buff|naked|nude|nudity|sex|sexy|sexual|erotic|fetish|porn|"
                    r"undress|lingerie|aroused|kink)")


# For tools only: "sex" is a normal parameter of medical calculators there.
EXPLICIT = re.compile(r"(?i)porn|erotic|\bnude|\bnaked|fetish|nsfw|onlyfans|\bescort")


def english(text):
    """ASCII letters and enough common English words (Malay or Indonesian in Latin script fail)."""
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(c.isascii() for c in letters) / len(letters) <= 0.97:
        return False
    words = re.findall(r"[a-z]+", text.lower())
    return bool(words) and sum(w in ENGLISH_WORDS for w in words) / len(words) >= 0.12


def ifeval_ids(root):
    """Instruction ids Google's IFEval verifiers can check (commented-out entries excluded)."""
    text = (paths(root)["ifeval"] / "instructions_registry.py").read_text()
    live = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    prefixes = dict(re.findall(r'^(_[A-Z_]+) = "([a-z_:-]+)"', live, re.M))
    return {prefixes[p] + n for p, n in re.findall(r'(_[A-Z_]+) \+ "([a-z_]+)"', live)}


def instruct_rows(funnel, rng, root):
    allowed = ifeval_ids(root)
    rows = []
    for r in load_dataset("allenai/IF_multi_constraints_upto5", split="train"):
        truth = r["ground_truth"]
        truth = (ast.literal_eval(truth) if isinstance(truth, str) else truth)[0]
        ids = truth["instruction_id"]
        if not set(ids) <= allowed:
            funnel("ifeval", "a constraint IFEval cannot check")
            continue
        if len(ids) > 3:
            funnel("ifeval", "more than 3 constraints (IFEval has 1-3)")
            continue
        if any(word in r["key"] for word in IF_EXCLUDED_SOURCES):
            funnel("ifeval", "base request from a math/code/safety/table/science set")
            continue
        messages = r["messages"]
        content = (ast.literal_eval(messages) if isinstance(messages, str) else messages)[0]["content"]
        base = content
        for constraint in r["constraint"].split("\t"):
            base = base.replace(constraint.strip(), " ")
        problem = base_request_problem(base)
        if problem:
            funnel("ifeval", problem)
            continue
        rows.append({"prompt": [{"role": "user", "content": content}], "task": "ifeval",
                     "source": f"if_multi/{len(ids)}", "answer": "",
                     "meta": {"instruction_id_list": ids,
                              "kwargs": [k or {} for k in truth["kwargs"]], "key": r["key"]},
                     "_compare": base})
    return rows + rlvr_rows(funnel, root)


# RLVR-IFeval (Tulu 3) writes one IFEval-template constraint per prompt and names its own
# checker. Only checkers that are an IFEval instruction with the same meaning are mapped:
# "around N" has no IFEval checker and "at most N" is IFEval's "less than N + 1"; its
# answer options, keyword counts and repeat-the-prompt rows differ from IFEval and are dropped.
RLVR_TO_IFEVAL = {
    "validate_lowercase": ("change_case:english_lowercase", {}),
    "validate_uppercase": ("change_case:english_capital", {}),
    "validate_frequency_capital_words": ("change_case:capital_word_frequency",
                                         {"N": "capital_frequency", "quantifier": "capital_relation"}),
    "validate_end": ("startend:end_checker", {"end_phrase": "end_phrase"}),
    "validate_quotation": ("startend:quotation", {}),
    "validate_sections": ("detectable_format:multiple_sections",
                          {"N": "num_sections", "section_splitter": "section_spliter"}),
    "validate_word_constraint": ("length_constraints:number_words",
                                 {"N": "num_words", "quantifier": "relation"}),
    "verify_sentence_constraint": ("length_constraints:number_sentences",
                                   {"N": "num_sentences", "quantifier": "relation"}),
    "validate_placeholders": ("detectable_content:number_placeholders", {"N": "num_placeholders"}),
    "validate_two_responses": ("combination:two_responses", {}),
    "validate_highlighted_sections": ("detectable_format:number_highlighted_sections",
                                      {"N": "num_highlights"}),
    "validate_json_format": ("detectable_format:json_format", {}),
    "validate_paragraphs": ("length_constraints:nth_paragraph_first_word",
                            {"N": "num_paragraphs", "i": "nth_paragraph", "first_word": "first_word"}),
    "verify_paragraph_count": ("length_constraints:number_paragraphs", {"N": "num_paragraphs"}),
    "verify_postscript": ("detectable_content:postscript", {"postscript_marker": "postscript_marker"}),
    "validate_no_commas": ("punctuation:no_comma", {}),
    "validate_forbidden_words": ("keywords:forbidden_words", {"forbidden_words": "forbidden_words"}),
    "verify_keywords": ("keywords:existence", {"keyword_list": "keywords"}),
    "validate_title": ("detectable_format:title", {}),
    "verify_bullet_points": ("detectable_format:number_bullet_lists", {"N": "num_bullets"}),
}


def ifeval_count_limits(root):
    """Largest value of each count argument in IFEval's own prompts (e.g. num_paragraphs 5)."""
    limits = defaultdict(int)
    for line in open(root / "questions.jsonl"):
        row = json.loads(line)
        if row["dataset"] != "ifeval":
            continue
        kwargs = row["kwargs"]
        for entry in (ast.literal_eval(kwargs) if isinstance(kwargs, str) else kwargs):
            for key, value in (entry or {}).items():
                if isinstance(value, int) and not isinstance(value, bool):
                    limits[key] = max(limits[key], value)
    return limits


def base_request_problem(base):
    """Why a base request does not look like an IFEval writing request, or None."""
    if not english(base) or "<unk>" in base:
        return "base request not English, or has <unk> (IFEval asks in English)"
    if SEXUAL.search(base):
        return "sexual content"
    if re.search(r"(?i)\b(python|javascript|sql|java|c\+\+) (function|code|program|script)|write a function", base):
        return "base request is a coding task"
    if re.search(r"\\\(|\\\[|\$[^$]+\$|\\frac|\^\d", base):
        return "base request is a math problem (LaTeX)"
    return None


def rlvr_rows(funnel, root):
    limits = ifeval_count_limits(root)
    rows = []
    for r in load_dataset("allenai/RLVR-IFeval", split="train"):
        truth = json.loads(r["ground_truth"])
        if truth["func_name"] not in RLVR_TO_IFEVAL:
            funnel("ifeval", "rlvr: constraint without an IFEval equivalent")
            continue
        instruction, names = RLVR_TO_IFEVAL[truth["func_name"]]
        kwargs = {ours: truth[theirs] for theirs, ours in names.items()}
        for key in ("capital_relation", "relation"):
            if key in kwargs:
                if kwargs[key] == "around":
                    kwargs = None
                    break
                if kwargs[key] == "at most":
                    count = {"capital_relation": "capital_frequency", "relation":
                             "num_words" if "num_words" in kwargs else "num_sentences"}[key]
                    kwargs[count] += 1
                    kwargs[key] = "less than"
        if kwargs is None:
            funnel("ifeval", "rlvr: 'around N' (no IFEval checker)")
            continue
        if any(isinstance(v, int) and v > limits.get(k, 0) for k, v in kwargs.items()):
            funnel("ifeval", "rlvr: count above any in IFEval (e.g. 30 paragraphs)")
            continue
        messages = r["messages"]
        content = (ast.literal_eval(messages) if isinstance(messages, str) else messages)[0]["content"]
        base = content.replace(r["constraint"].strip(), " ")
        problem = base_request_problem(base)
        if problem:
            funnel("ifeval", problem)
            continue
        rows.append({"prompt": [{"role": "user", "content": content}], "task": "ifeval",
                     "source": "rlvr_ifeval/1", "answer": "",
                     "meta": {"instruction_id_list": [instruction], "kwargs": [kwargs],
                              "key": truth["func_name"]},
                     "_compare": base})
    return rows


# ---------------------------------------------------------------- math

NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")


def math_rows(funnel, rng):
    rows = []
    for r in load_dataset("openai/gsm8k", "main", split="train"):
        answer = r["answer"].split("####")[-1].strip().replace(",", "")
        if not NUMBER.match(answer):
            funnel("math", "gsm8k: answer not a plain number")
            continue
        rows.append({"prompt": [{"role": "user", "content": r["question"].strip() + NUMERIC_INSTRUCTION}],
                     "task": "math", "source": "gsm8k", "answer": answer, "meta": {},
                     "_compare": r["question"]})
    for r in load_dataset("meta-math/MetaMathQA", split="train"):
        if r["type"] not in ("GSM_SV", "GSM_FOBAR"):
            continue
        answer = r["response"].rsplit("The answer is:", 1)[-1].strip().replace(",", "")
        if not NUMBER.match(answer):
            funnel("math", "metamath: answer not a plain number")
            continue
        query = r["query"].strip()
        doubled = "What is the value of unknown variable x? What is the value of unknown variable x?"
        query = query.replace(doubled, "What is the value of unknown variable x?")
        rows.append({"prompt": [{"role": "user", "content": query + NUMERIC_INSTRUCTION}],
                     "task": "math", "source": f"metamath/{r['type']}", "answer": answer, "meta": {},
                     "_compare": query})
    return rows


# ---------------------------------------------------------------- selection

QUOTAS = {  # share of the domain's rows per source group
    "code": {"kodcode/Filter": 0.4, "kodcode/Prefill": 0.3, "kodcode/Leetcode": 0.15,
             "kodcode/Evol": 0.15},
    "tools": {"toolace/simple": 0.4, "toolace/multiple": 0.3, "toolace/decline": 0.05,
              "xlam/irrelevance": 0.25},
    "ifeval": {"ifeval/1": 0.56, "ifeval/2": 0.33, "ifeval/3": 0.11},  # by constraint count
    "math": {"gsm8k": 0.7, "metamath/GSM_SV": 0.15, "metamath/GSM_FOBAR": 0.15},
}


def target_of(row):
    if row["task"] == "ifeval":
        return f"ifeval/{len(row['meta']['instruction_id_list'])}"
    return TARGET.get(row["source"]) or TARGET[row["task"]]


def quota_group(row):
    return target_of(row) if row["task"] == "ifeval" else row["source"]


def deciles(lengths):
    lengths = sorted(lengths)
    return [lengths[round(i * (len(lengths) - 1) / LENGTH_BINS)] for i in range(LENGTH_BINS + 1)]


def length_matched(rows, edges, n, rng, spill=True):
    """Up to n rows, as even as the pool allows over the target's length deciles.

    With spill, a decile the pool cannot fill is topped up from the nearest deciles that have
    rows left; without, the shortfall is returned to the caller (fewer than n rows).
    """
    bins = [[] for _ in range(LENGTH_BINS)]
    for row in rows:
        bins[min(bisect_right(edges, row["prompt_tokens"]) - 1, LENGTH_BINS - 1)].append(row)
    for group in bins:
        rng.shuffle(group)
    want = [n // LENGTH_BINS + (i < n % LENGTH_BINS) for i in range(LENGTH_BINS)]
    chosen = []
    for i in range(LENGTH_BINS):
        take = min(want[i], len(bins[i]))
        chosen += bins[i][:take]
        bins[i] = bins[i][take:]
        want[i] -= take
    for i in range(LENGTH_BINS if spill else 0):
        for j in sorted(range(LENGTH_BINS), key=lambda j: abs(j - i)):
            take = min(want[i], len(bins[j]))
            chosen += bins[j][:take]
            bins[j] = bins[j][take:]
            want[i] -= take
    return chosen


def select(rows, domain, total, edges, rng):
    by_group = defaultdict(list)
    for row in rows:
        by_group[quota_group(row)].append(row)
    chosen, short = [], 0
    for name, group in sorted(by_group.items()):
        take = round(total * QUOTAS[domain].get(name, 0))
        picked = length_matched(group, edges[target_of(group[0])], take, rng, spill=False)
        short += take - len(picked)
        chosen += picked
    if short > 0:  # groups short in some deciles: fill from the rest of the same benchmark split,
        # strictly length-matched too, so a domain may end with fewer rows than asked for
        taken = {id(row) for row in chosen}
        rest = defaultdict(list)
        for row in rows:
            if id(row) not in taken:
                rest[target_of(row)].append(row)
        for target, group in sorted(rest.items(), key=lambda item: -len(item[1])):
            if short <= 0:
                break
            picked = length_matched(group, edges[target], short, rng, spill=False)
            chosen += picked
            short -= len(picked)
    return chosen


def main():
    arguments = parse_arguments()
    rng = random.Random(arguments.seed)
    root = Path("questions").resolve()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    eval_grams, eval_lengths = defaultdict(set), defaultdict(list)
    for line in open(root / "questions.jsonl"):
        row = json.loads(line)
        text = row.get("original_prompt") or row["messages"][-1]["content"]
        eval_grams[row["dataset"]] |= grams(text)
        length = prompt_tokens(tokenizer, row["messages"])
        eval_lengths[row["dataset"]].append(length)
        if row["dataset"] == "ifeval":
            eval_lengths[f"ifeval/{len(ast.literal_eval(str(row['instruction_id_list'])))}"].append(length)
    edges = {name: deciles(lengths) for name, lengths in eval_lengths.items()}
    every_eval_gram = set().union(*eval_grams.values())

    checker = load_bfcl(paths(root)["bfcl"])
    funnel = Funnel()
    built = {"code": code_rows(funnel, rng), "tools": tools_rows(funnel, rng, paths(root)["bfcl"]),
             "ifeval": instruct_rows(funnel, rng, root), "math": math_rows(funnel, rng)}

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for domain, rows in built.items():
        clean, seen = [], set()
        for row in rows:
            key = row["_compare"].strip().lower()
            if key in seen:
                funnel(domain, "duplicate prompt")
                continue
            seen.add(key)
            overlap = grams(row["_compare"]) & every_eval_gram
            if overlap:
                funnel(domain, "8-gram overlap with an eval prompt")
                row["_overlap"] = " ".join(next(iter(overlap)))
                continue
            row["prompt_tokens"] = prompt_tokens(tokenizer, row["prompt"])
            low, high = edges[target_of(row)][0], edges[target_of(row)][-1]
            if not 0.9 * low <= row["prompt_tokens"] <= 1.1 * high:
                funnel(domain, "prompt length outside the benchmark's range (+-10%)")
                continue
            if domain == "tools" and not gold_call_passes(row, checker):
                funnel(domain, "its own gold call fails BFCL's checker (types, nesting)")
                continue
            clean.append(row)
        chosen = select(clean, domain, arguments.per_domain, edges, rng)
        if domain == "code":  # running ~30k programs for the whole pool is not worth it
            from behaviour_check import run_tests
            verdicts = run_tests([{"code": r["meta"]["solution"], "test": r["meta"]["test"]} for r in chosen])
            for row, verdict in zip(chosen, verdicts):
                if verdict != "pass":
                    funnel(domain, f"reference solution does not pass its tests here ({verdict}), after selection")
            chosen = [row for row, verdict in zip(chosen, verdicts) if verdict == "pass"]
        for index, row in enumerate(chosen):
            row["qid"] = index
            row.pop("_compare", None)
        with (out / f"{domain}.jsonl").open("w") as stream:
            for row in chosen:
                stream.write(json.dumps(row) + "\n")
        summary[domain] = {"built": len(rows), "clean": len(clean), "chosen": chosen, "eval_lengths": eval_lengths,
                           "dropped": dict(funnel.counts[domain]),
                           "overlaps": [r["_overlap"] for r in rows if "_overlap" in r][:8]}
        print(f"{domain}: {len(rows)} built, {len(clean)} clean, {len(chosen)} written", flush=True)
    write_review(summary, arguments.review, rng)


def gold_call_passes(row, checker):
    """Every tools row must be gradable: its own reference call has to pass BFCL's AST checker."""
    category = row["meta"]["category"]
    if category not in ("simple", "multiple"):
        return True
    bfcl_row = {"function": row["meta"]["function"], "reference": row["meta"]["reference"],
                "category": "simple_python" if category == "simple" else category}
    try:
        return grade_call(bfcl_row, row["answer"], *checker)[0]
    except Exception:
        return False


def prompt_tokens(tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def percentiles(lengths):
    lengths = sorted(lengths)
    return " / ".join(str(lengths[round(p * (len(lengths) - 1))]) for p in (0.1, 0.25, 0.5, 0.75, 0.9))


def write_review(summary, path, rng):
    lines = ["# Base training prompts per domain (`prepare_domains.py`)", "",
             "Prompts only: no model output yet. Each row keeps its grader data in `meta`.", ""]
    lines += ["| domain | rows built | after filters | written | mean / max prompt tokens |",
              "|---|---:|---:|---:|---:|"]
    for domain, s in summary.items():
        tokens = [r["prompt_tokens"] for r in s["chosen"]]
        lines.append(f"| {domain} | {s['built']} | {s['clean']} | {len(s['chosen'])} | "
                     f"{sum(tokens) / len(tokens):.0f} / {max(tokens)} |")
    for domain, s in summary.items():
        lines += ["", f"## {domain}", "", "Dropped:", ""]
        lines += [f"- {reason}: {count}" for reason, count in sorted(s["dropped"].items(), key=lambda x: -x[1])]
        if s["overlaps"]:
            lines += ["", "Example 8-grams that caused a drop: " + "; ".join(f"`{o}`" for o in s["overlaps"])]
        lines += ["", "Prompt tokens, p10 / p25 / p50 / p75 / p90 (written rows vs their benchmark):", "",
                  "| group | n | written | benchmark |", "|---|---:|---|---|"]
        groups = defaultdict(list)
        for row in s["chosen"]:
            groups[(row["source"], target_of(row))].append(row["prompt_tokens"])
        for (source, target), lengths in sorted(groups.items()):
            lines.append(f"| {source} vs {target} | {len(lengths)} | {percentiles(lengths)} | "
                         f"{percentiles(s['eval_lengths'][target])} |")
        lines += ["", "Written, by source:", ""]
        lines += [f"- {source}: {count}" for source, count in
                  sorted(Counter(r["source"] for r in s["chosen"]).items())]
        if domain == "ifeval":
            ids = Counter(i for r in s["chosen"] for i in r["meta"]["instruction_id_list"])
            lines += ["", "Instruction ids (all rows): " + ", ".join(f"{k} {v}" for k, v in ids.most_common())]
        lines += ["", "Random examples (one per source):", ""]
        by_source = defaultdict(list)
        for row in s["chosen"]:
            by_source[row["source"]].append(row)
        for source, group in sorted(by_source.items()):
            row = rng.choice(group)
            lines += [f"### {source} (qid {row['qid']}, {row['prompt_tokens']} prompt tokens)", ""]
            for message in row["prompt"]:
                content = message["content"]
                if message["role"] == "system":
                    names = [f["name"] for f in row["meta"]["function"]]
                    content = f"[BFCL system prompt with {len(names)} function(s): {', '.join(names)}]"
                lines += [f"**{message['role']}:**", "", "```text", content.strip(), "```", ""]
            if row["task"] == "math":
                lines += [f"answer: `{row['answer']}`", ""]
            elif row["task"] == "tools":
                lines += [f"reference: `{json.dumps(row['meta']['reference'])}`  original answer: `{row['answer'][:200]}`", ""]
            elif row["task"] == "ifeval":
                lines += [f"instructions: `{row['meta']['instruction_id_list']}` kwargs: `{json.dumps(row['meta']['kwargs'])}`", ""]
            elif row["task"] == "code":
                lines += [f"tests ({row['meta']['test'].count('def test')} test functions), difficulty {row['meta']['difficulty']}", ""]
    Path(path).write_text("\n".join(lines) + "\n")
    print(f"review -> {path}")


if __name__ == "__main__":
    main()
