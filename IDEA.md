# Shorter thinking for Qwen3-0.6B by preference learning on its own samples

**Result (full evaluation, 8 samples):** 47.0% fewer generated tokens [CI +45.8, +48.1] at
−0.78 pp accuracy [CI −1.49, −0.05]. The model is a LoRA adapter (`adapters/Db`), and the
evaluation output is `runs/Db-s8/results.json`.

| benchmark | base acc / tokens | ours acc / tokens | tokens saved | accuracy change |
|---|---|---|---|---|
| BFCL irrelevance | 82.8% / 214 | 88.9% / 115 | +46.1% | +6.04 pp |
| BFCL multiple | 76.6% / 224 | 75.4% / 130 | +41.9% | −1.12 pp |
| BFCL simple | 82.2% / 193 | 80.8% / 125 | +35.3% | −1.41 pp |
| IFEval | 57.8% / 694 | 57.5% / 314 | +54.7% | −0.28 pp |
| MBPP+ | 52.1% / 2,115 | 50.8% / 1,418 | +33.0% | −1.22 pp |
| SVAMP | 88.4% / 684 | 86.0% / 298 | +56.4% | −2.41 pp |
| **average (4 benchmarks)** | | | **+47.0%** | **−0.78 pp** |

## Idea in one paragraph

Let the model show you what it already does well, then prefer it. Sample Qwen3-0.6B 4 times
on about 11k training prompts that look like the benchmarks but are not them. Grade every
sample with the evaluation's own graders. Then train with DPO: the shortest correct sample
is preferred over (a) a wrong or truncated sample and (b) a much longer correct sample. No
text is written or edited by another model. Everything the adapter learns to say, the base
model already said.

## Motivation: what failed first

The first attempts were supervised fine-tuning, on GSM8K / ARC / CommonsenseQA traces and
later on benchmark-like prompts (details in `reports/FINDINGS.md`, sections 2–15):

- **Cut each trace at the first committed answer** (a gold-free stopping cue found in the
  model's own text) and fine-tune on that. This saved 35% on SVAMP and nothing elsewhere,
  and long outputs on IFEval and MBPP+ looped more often.
- **Edit the traces by rules** (remove filler, copies of the question, format talk,
  articles), in the style of ThinkingCap-27B's traces. This failed badly: −6 to −10 pp
  accuracy and 2–6× more runaway loops. For a 0.6B model, training on text it would not
  write itself damages its ability to stop. **Compress by selection, not by rewriting.**
- **Fine-tune on the shortest correct own sample:** about +5% tokens saved, no accuracy
  cost. That is safe, but too weak: imitating samples the model already produces changes
  little.

The missing ingredient was a negative signal: something telling the model what not to write.

## Method

1. **Training prompts matched to each benchmark** (`prepare_domains.py`), none from the
   benchmarks:

   | domain | rows | sources | matched to |
   |---|---:|---|---|
   | code | 2,989 | KodCode-V1 | MBPP+ |
   | tools | 2,986 | ToolACE, xlam irrelevance | BFCL |
   | ifeval | 2,385 | IF_multi_constraints, RLVR-IFeval | IFEval |
   | math | 2,967 | GSM8K, MetaMathQA | SVAMP |

   - Each prompt is written in its benchmark's format (prompt template, system prompt,
     answer instruction).
   - Prompt length is sampled per decile of the benchmark's own length distribution.
   - Any prompt sharing an 8-gram with any eval prompt is dropped.
   - Only rows the eval graders can score are kept: the KodCode reference solution must
     pass its tests, and the ToolACE gold call must pass BFCL's checker.
   - A behaviour check confirms that the base model's accuracy and length on these prompts
     are close to the benchmarks' (for example, code 53.8% / 972 median tokens vs MBPP+
     51.5% / 1,019).
2. **Sampling and grading** (`behaviour_check.py`): 4 samples per prompt with the eval's
   sampling settings and token caps (45,308 samples, about 25 h on this GPU). Each sample is
   graded with the eval's own graders, and KodCode's pytest tests are run for code.
3. **Preference pairs** (`train_dpo_offline.py --balance`):
   - chosen = the shortest correct, untruncated sample;
   - rejected = the longest wrong or truncated sample ("wrong pair"), and/or the longest
     correct sample if it is at least 1.5× longer ("long pair");
   - per domain, long pairs are capped at the number of wrong pairs, giving 5,862 pairs.
4. **DPO:** sigmoid loss, β = 0.1, on sequence log-probs of the stored token ids. The
   reference model is the base (the adapter switched off). LoRA r = 32, alpha 64, all
   linear layers, lr 2e-5 (cosine), 16 pairs per step, 1 epoch (367 steps, 1.9 h).

## What we learned (all from the ablations; `--samples 2` unless noted)

| variant | tokens saved | accuracy change |
|---|---|---|
| SFT on the shortest correct own sample | +4.5% | +0.90 pp |
| Offline GRPO on the same samples (length-penalised reward) | −13.8% | −9.41 pp |
| DPO, wrong pairs only | +23.0% | +0.81 pp |
| DPO, long pairs only | +83.6% | −13.04 pp |
| DPO, both pair kinds (unbalanced) | +50.2% | −0.41 pp |
| **DPO, both, balanced per domain (8 samples)** | **+47.0%** | **−0.78 pp** |

- **The two pair kinds do different jobs.** Long pairs drive the compression; on their own
  they over-compress and accuracy collapses everywhere. Wrong pairs keep accuracy, and also
  shorten outputs, because wrong samples are often long or looping. The mix works.
- **Without balancing, a domain with few wrong samples loses accuracy.** The base model is
  right on about 85% of math prompts, so math got many long pairs and few wrong ones, and
  SVAMP lost 3.9 pp. Capping long pairs at the number of wrong pairs reduced that loss to
  2.4 pp. This is measured with different sample counts, so it is indicative only.
- **Offline GRPO on stored samples failed.** With no fresh rollouts, the policy drifts and
  about half of all tokens end up clipped. The negative push on long samples produced more
  runaway loops, not fewer. DPO on the same data does not have this problem.
- **The model still reasons.** Median thinking roughly halves (SVAMP 382 → 197 tokens), and
  none of the 22,072 evaluation outputs thinks for fewer than 20 tokens. Truncation at the token cap goes down (MBPP+ 7.3% →
  3.8%).

**Caveat:** the final variant (balancing) was chosen after seeing the unbalanced variant's
results on the same benchmark questions. The full evaluation uses 6 fresh samples per
question but the same questions, so it is slightly optimistic.

## Possible next steps

- Iterate: sample again from the trained model and run a second DPO round. This is the
  cheap, offline approximation of online RL.
- Add an SFT term on the chosen samples (RPO) to recover SVAMP.
- Online GRPO with fresh rollouts, if more compute is available.

## Reproduce

```bash
python prepare_domains.py --out data/domains --per-domain 3000
python behaviour_check.py sample --per-domain 0 --samples 4 --gpu-memory-utilization 0.8 --out samples/domains.jsonl
python behaviour_check.py grade samples/domains.jsonl --base runs/base-s2 --report reports/domains_samples.md
python train_dpo_offline.py samples/domains.graded.jsonl --balance --out adapters/Db
python evaluate.py --gpu-memory-utilization 0.8 --adapter adapters/Db --out runs/Db-s8
python evaluate.py --gpu-memory-utilization 0.8 --out runs/base-s8
python compare.py runs/base-s8 runs/Db-s8
```

## Hardware and cost

- **Machine:** one NVIDIA RTX 2080 Ti (11 GB, Turing, so fp16 instead of bf16 and vLLM's
  Triton attention), Ryzen 9 3900X, 32 GB RAM.
- **Settings:** the evaluation settings are unchanged. `--gpu-memory-utilization 0.8`
  (0.78 on retry) only sizes vLLM's KV cache.
- **Wall-clock time:** sampling about 25 h; DPO training about 2 h; a full 8-sample
  evaluation about 7 h (adapter) and 10 h (base).
- **API spend:** about $0.01, on one abandoned attempt to have Claude edit traces (every
  request was refused).
- **Disclosure:** ThinkingCap-27B was run locally (llama.cpp) only to study how its traces
  differ from its base model's. No ThinkingCap output is used in training.

The full experiment log, including negative results, is in `reports/FINDINGS.md`.
