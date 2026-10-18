"""LoRA SFT of Qwen3-0.6B on an arm file built by build_arms.py.

    python train_sft.py data/arms/T.jsonl --out adapters/T

The loss is on the completion only (<think>...</think> and the answer, then <|im_end|>).
The prompt is the same chat template the evaluation uses (thinking on). On a Turing GPU
(no bf16) the base weights stay fp32 and the forward pass runs under fp16 autocast.
The adapter is saved in PEFT format, which evaluate.py --adapter loads.
"""

import argparse
import json
import math
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

MODEL = "Qwen/Qwen3-0.6B"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arm")
    parser.add_argument("--out", required=True)
    parser.add_argument("--keep", help="JSONL of {qid} rows to keep (e.g. the fidelity filter's output)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--batch-tokens", type=int, default=4096, help="Tokens per micro-batch")
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def encode(tokenizer, row, max_length):
    prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True)
    completion = row["completion"][0]["content"] + "<|im_end|>\n"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    ids = prompt_ids + completion_ids
    if len(ids) > max_length:
        return None
    return {"ids": ids, "labels": [-100] * len(prompt_ids) + completion_ids}


def batches(examples, batch_tokens, rng):
    """Micro-batches of similar length, padded size at most batch_tokens."""
    order = sorted(examples, key=lambda e: len(e["ids"]))
    out, current = [], []
    for example in order:
        longest = max([len(e["ids"]) for e in current] + [len(example["ids"])])
        if current and longest * (len(current) + 1) > batch_tokens:
            out.append(current)
            current = []
        current.append(example)
    if current:
        out.append(current)
    rng.shuffle(out)
    return out


def collate(batch, pad_id, device):
    longest = max(len(e["ids"]) for e in batch)
    ids = torch.full((len(batch), longest), pad_id)
    labels = torch.full((len(batch), longest), -100)
    mask = torch.zeros((len(batch), longest), dtype=torch.long)
    for i, e in enumerate(batch):
        ids[i, :len(e["ids"])] = torch.tensor(e["ids"])
        labels[i, :len(e["labels"])] = torch.tensor(e["labels"])
        mask[i, :len(e["ids"])] = 1
    return ids.to(device), labels.to(device), mask.to(device)


def completion_loss(model, ids, labels, mask, chunk=1024):
    """Mean cross-entropy over completion tokens, without materializing full logits.

    The transformer body runs once; hidden states are kept only where a label exists,
    and the lm_head + loss run in checkpointed chunks (152k-word vocabulary).
    """
    inner = model.get_base_model()
    hidden = inner.model(input_ids=ids, attention_mask=mask).last_hidden_state
    targets = labels[:, 1:]
    keep = targets != -100
    hidden = hidden[:, :-1][keep]
    targets = targets[keep]

    def chunk_loss(h, t):
        return F.cross_entropy(inner.lm_head(h).float(), t, reduction="sum")

    total = sum(checkpoint(chunk_loss, hidden[i:i + chunk], targets[i:i + chunk], use_reentrant=False)
                for i in range(0, len(targets), chunk))
    return total / max(1, len(targets))


def main():
    arguments = parse_arguments()
    rng = random.Random(arguments.seed)
    torch.manual_seed(arguments.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(line) for line in open(arguments.arm)]
    if arguments.keep:
        keep = {json.loads(line)["qid"] for line in open(arguments.keep)}
        rows = [row for row in rows if row["qid"] in keep]
    examples = [e for row in rows if (e := encode(tokenizer, row, arguments.max_length))]
    print(f"{len(examples)} of {len(rows)} rows fit {arguments.max_length} tokens", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=arguments.rank, lora_alpha=2 * arguments.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(batches(examples, arguments.batch_tokens, rng)) / arguments.accumulate)
    total = steps_per_epoch * arguments.epochs
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=arguments.lr, weight_decay=0.0)
    schedule = get_cosine_schedule_with_warmup(optimizer, max(1, total // 20), total)
    scaler = torch.amp.GradScaler("cuda")

    model.train()
    step, start = 0, time.time()
    for epoch in range(arguments.epochs):
        micro = batches(examples, arguments.batch_tokens, rng)
        running, count = 0.0, 0
        for index, batch in enumerate(micro):
            ids, labels, mask = collate(batch, tokenizer.pad_token_id, "cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = completion_loss(model, ids, labels, mask)
            scaler.scale(loss / arguments.accumulate).backward()
            running += loss.item()
            count += 1
            if (index + 1) % arguments.accumulate == 0 or index + 1 == len(micro):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                schedule.step()
                step += 1
                if step % 10 == 0 or step == total:
                    print(f"epoch {epoch} step {step}/{total}  loss {running / count:.4f}  "
                          f"lr {schedule.get_last_lr()[0]:.2e}  {(time.time() - start) / 60:.1f} min",
                          flush=True)
                    running, count = 0.0, 0
    model.save_pretrained(arguments.out)
    tokenizer.save_pretrained(arguments.out)
    print(f"saved {arguments.out}", flush=True)


if __name__ == "__main__":
    main()
