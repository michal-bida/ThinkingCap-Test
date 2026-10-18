"""Arm D: DPO on pairs of the base model's stored samples (no generation).

    python train_dpo_offline.py samples/domains.graded.jsonl --out adapters/D

Same data as arm G1 (train_grpo_offline.py), a different objective, so D vs G1 isolates the
objective. Per prompt with a correct, untruncated sample, chosen = the shortest of those.
Up to two pairs per prompt, each against the same chosen sample:
  - rejected = a wrong or truncated sample (the longest such), if there is one;
  - rejected = the longest correct sample, if it is at least --long-ratio times longer.
Samples are used token for token as stored. Loss is standard DPO (sigmoid, beta 0.1) on
sequence log-probabilities; the reference model is the base, i.e. the same network with the
adapter switched off, as in G1. No extra SFT term, so the comparison with G1 stays clean.
"""

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from train_grpo_offline import DOMAINS, MODEL, token_logprobs


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples")
    parser.add_argument("--data", default="data/domains")
    parser.add_argument("--out", required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--long-ratio", type=float, default=1.5)
    parser.add_argument("--kinds", default="wrong,long",
                        help="Rejected kinds to pair against: wrong (wrong or truncated), long (longer correct)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--accumulate", type=int, default=16, help="Pairs per optimizer step")
    parser.add_argument("--max-length", type=int, default=8704, help="Pairs with a longer side are skipped")
    parser.add_argument("--balance", action="store_true",
                        help="Per domain, keep at most as many long pairs as wrong pairs (random subset)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def pairs(arguments, tokenizer):
    prompts = {}
    for domain in DOMAINS:
        for line in open(Path(arguments.data) / f"{domain}.jsonl"):
            row = json.loads(line)
            prompts[(domain, row["qid"])] = tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True, enable_thinking=True)
    groups = defaultdict(list)
    for line in open(arguments.samples):
        sample = json.loads(line)
        if sample.get("correct") is not None:
            groups[(sample["task"], sample["qid"])].append(sample)
    out, kinds, skipped = [], Counter(), 0
    for key, group in groups.items():
        good = [s for s in group if s["correct"] and not s["truncated"]]
        if not good:
            continue
        chosen = min(good, key=lambda s: s["output_tokens"])
        bad = [s for s in group if not (s["correct"] and not s["truncated"])]
        candidates = []
        kinds_wanted = arguments.kinds.split(",")
        if bad and "wrong" in kinds_wanted:
            candidates.append(("wrong", max(bad, key=lambda s: s["output_tokens"])))
        longest = max(good, key=lambda s: s["output_tokens"])
        if "long" in kinds_wanted and longest["output_tokens"] >= arguments.long_ratio * chosen["output_tokens"]:
            candidates.append(("long", longest))
        prompt = prompts[key]
        for kind, rejected in candidates:
            if len(prompt) + max(len(chosen["token_ids"]), len(rejected["token_ids"])) > arguments.max_length:
                skipped += 1
                continue
            out.append({"domain": key[0], "kind": kind, "prompt": prompt,
                        "chosen": chosen["token_ids"], "rejected": rejected["token_ids"]})
    if arguments.balance:
        # Ablations (Dw, Dl) showed long pairs drive the compression and wrong pairs hold accuracy;
        # math had twice as many long as wrong pairs and was the one benchmark D lost accuracy on.
        rng = random.Random(arguments.seed)
        kept = []
        for domain in DOMAINS:
            wrong = [p for p in out if p["domain"] == domain and p["kind"] == "wrong"]
            long = [p for p in out if p["domain"] == domain and p["kind"] == "long"]
            kept += wrong + (rng.sample(long, len(wrong)) if len(long) > len(wrong) else long)
        out = kept
    for pair in out:
        kinds[(pair["domain"], pair["kind"])] += 1
    print(f"{len(out)} pairs, {skipped} skipped as too long; per (domain, rejected kind): "
          f"{dict(sorted(kinds.items()))}", flush=True)
    return out


def sequence_logprob(inner, prompt, completion):
    ids = torch.tensor([prompt + completion], device="cuda")
    labels = torch.tensor([[-100] * len(prompt) + completion], device="cuda")
    logprobs, _ = token_logprobs(inner, ids, None, labels)  # one unpadded row: no mask needed
    return logprobs.sum()


def main():
    arguments = parse_arguments()
    rng = random.Random(arguments.seed)
    torch.manual_seed(arguments.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    data = pairs(arguments, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=arguments.rank, lora_alpha=2 * arguments.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    inner = model.get_base_model()

    total = math.ceil(len(data) / arguments.accumulate) * arguments.epochs
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=arguments.lr, weight_decay=0.0)
    schedule = get_cosine_schedule_with_warmup(optimizer, max(1, total // 20), total)
    scaler = torch.amp.GradScaler("cuda")

    model.train()
    step, start = 0, time.time()
    for epoch in range(arguments.epochs):
        rng.shuffle(data)
        running = defaultdict(float)
        for index, pair in enumerate(data):
            with torch.autocast("cuda", dtype=torch.float16):
                with torch.no_grad(), model.disable_adapter():
                    reference = [sequence_logprob(inner, pair["prompt"], pair[side]) for side in ("chosen", "rejected")]
                policy = [sequence_logprob(inner, pair["prompt"], pair[side]) for side in ("chosen", "rejected")]
            chosen_reward = arguments.beta * (policy[0] - reference[0]).float()
            rejected_reward = arguments.beta * (policy[1] - reference[1]).float()
            loss = -F.logsigmoid(chosen_reward - rejected_reward)
            scaler.scale(loss / arguments.accumulate).backward()
            running["loss"] += loss.item()
            running["chosen"] += chosen_reward.item()
            running["rejected"] += rejected_reward.item()
            running["accuracy"] += float(chosen_reward.item() > rejected_reward.item())
            running["count"] += 1
            if (index + 1) % arguments.accumulate == 0 or index + 1 == len(data):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                schedule.step()
                step += 1
                if step % 10 == 0 or step == total:
                    n = running["count"]
                    print(f"epoch {epoch} step {step}/{total}  loss {running['loss'] / n:.4f}  "
                          f"reward chosen {running['chosen'] / n:+.3f} rejected {running['rejected'] / n:+.3f}  "
                          f"pair accuracy {100 * running['accuracy'] / n:.0f}%  "
                          f"lr {schedule.get_last_lr()[0]:.2e}  {(time.time() - start) / 60:.1f} min",
                          flush=True)
                    running = defaultdict(float)
    model.save_pretrained(arguments.out)
    tokenizer.save_pretrained(arguments.out)
    print(f"saved {arguments.out}", flush=True)


if __name__ == "__main__":
    main()
