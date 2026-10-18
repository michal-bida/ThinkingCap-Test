# Findings and plan (written while you were away)

Running log: newest entries are appended at the end.

## 1. Retraction: the "cut at first gold mention" test

The forced-answer test (`reports/forced_answer_pilot.txt`) chose the cut with the gold
answer, so "accuracy is kept" is true by construction. It says nothing about whether a
model can learn *when* to stop. It is no longer used as evidence. The data arm built from
it (`data/arms/B.jsonl`) is dropped for the same reason.

## 2. A stopping cue that needs no gold answer (`analyze_cues.py`, `reports/cues_train3k.txt`)

Question: is there something in the model's *own* text that marks where the useful
reasoning ends? Candidate: the first time it **commits** to an answer ("So the answer is
12.", not "maybe the answer is E?"). Gold is only used afterwards, to score.

Qwen3-0.6B, 3,000 training questions × 4 samples, finished traces:

| source | first committed answer right | final answer right | answer changes after it | thinking after it |
|---|---:|---:|---:|---:|
| ARC-Easy | 83.5% | 84.4% | 6.9% | 14% (median 32 tok) |
| CommonsenseQA | 54.4% | 55.1% | 7.1% | 15% (median 32 tok) |
| GSM8K | 88.8% | 89.4% | 3.5% | **43% (median 494 tok)** |

- After its first committed answer, the model almost never changes its mind. On GSM8K,
  re-checking fixes 1.9% of traces and breaks 1.3%. 31 of those 39 breaks are the model
  copying the prompt's format example ("Final answer: 42").
- Traces that hit the 4,096-token cap are scored 0. That is 22% of GSM8K samples, and the
  eval uses the same cap. 78% of them had already committed to an answer (median token
  1,691), and that answer was right 56% of the time. Stopping at the first committed
  answer would save tokens and also recover about 9 points of GSM8K accuracy.
- I checked the extraction by hand and fixed three bugs before these numbers: running
  totals counted as answers; "Final Answer:" followed by a `\boxed{}` read as empty; the
  "42" format example counted as a candidate.

So the learnable cue is: **once you state an answer without hedging, end the thinking.**
Hedged statements ("maybe E", "A or C") are not commitments. The model can tell the two
apart from its own wording, with no gold needed.

## 3. Synonyms / word cost

Qwen's tokenizer already spends 1.05 tokens per word on these traces. Even if every
multi-token word became a one-token word, the saving would be at most 4.8%, and most of
that is contractions (that's, it's, let's = 2 tokens each). Not worth a method. Dropping
articles and filler, and above all whole passages, matters far more.

## 4. ThinkingCap-27B vs base Qwen3.8-27B (`reports/compare_models.md`)

Same 300 training questions, sample 0, lengths in Qwen3-0.6B tokens:

| source | Qwen3.8 thinking | ThinkingCap thinking | Qwen3.8 correct | ThinkingCap correct |
|---|---:|---:|---:|---:|
| ARC-Easy | 117 | 66 | 99% | 99% |
| CommonsenseQA | 570 (median 150) | 88 (median 57) | 87% | 84% |
| GSM8K | 230 | 105 | 95% | 96% |
| Skywork | 2,061 | 1,105 | 81% | 87% |

Reading paired traces (both correct, base much longer):

1. **The terse, article-less register is not ThinkingCap's doing.** Base Qwen3.8 already
   writes "We need answer simple multiple choice." ThinkingCap kept the register and cut
   the amount.
2. **What ThinkingCap removes:**
   - copying the question and options back into the thinking (base: "Question: ...
     Options: A ... B ...");
   - wandering over alternative readings ("Maybe a riddle? In HTML? Nautical terms?");
   - re-checking after the answer is found ("Double-check", "Is that too easy?", "Let's
     consider if...");
   - talk about the output ("desired oververbosity 9", "Ensure final line exactly");
   - a worked solution in the answer section repeating the thinking: GSM8K answer section
     60 → 24 tokens; ThinkingCap answers just "Final answer: X".
3. **What it keeps:** the one decisive fact or computation, a one-clause justification,
   and on harder problems a single targeted check ("Check right triangle: 30² = h² + 15²").
4. **Structure:** every ThinkingCap trace opens by labelling the task and its difficulty
   ("We need answer simple arithmetic.") and ends right after the first committed answer
   ("... =13 gave sister."). That is the same cue as section 2, found independently in the
   0.6B's own traces.

ThinkingCap's gold-commit position is 86% of the thinking on GSM8K (base 70%, 0.6B 33%).
In other words, almost nothing follows the answer.

## 5. Plan for the autonomous experiments

Goal: test whether the cue in section 2, plus ThinkingCap-style removals, can be trained
into Qwen3-0.6B, and whether that carries over to the eval benchmarks (code, function
calling, instruction following and arithmetic: none of which are in the training data).

Training data comes from the 0.6B's **own** correct traces (2,398 questions, ARC / CSQA /
GSM8K), so the model is not taught reasoning it cannot produce. Two arms:

- **S, a rule-based cut.** The shortest correct trace, with its thinking cut after the
  sentence holding the first *committed* answer statement. The same rule as section 2:
  no gold involved in choosing the cut. The answer section is kept verbatim.
- **R, an LLM edit.** Claude edits the same thinking **by deletion and condensation
  only**, guided by the removal list in section 4:
  - no restating the question;
  - no format talk;
  - no ambiguity wandering;
  - stop at the first committed answer;
  - at most one targeted check on multi-step problems;
  - drop articles and filler.

  It may not add new reasoning. Each edit is then checked with the 0.6B: given the edited
  thinking, it must still produce the answer in at least 3 of 4 samples. This is a
  fidelity filter for training data. It is not an accuracy result.

Both are trained as LoRA adapters (same hyperparameters, same questions) and evaluated
with `evaluate.py --samples 2` against a base run with `--samples 2`, using `compare.py`.
`--samples 2` is a dev-level check; only 8 samples count as the full evaluation.

Budget: the Claude edits for about 2,400 traces cost an estimated $15–20 (assuming
Sonnet-class pricing). I'll pilot on 20 traces and read them before doing the rest.

GPU order (one job at a time): Qwen3.8 low effort (running) → base eval with 2 samples
→ fidelity check of the R edits → train S → train R → eval S → eval R.

## 6. Update: the Claude-edit arm (R) is dropped

Every edit request to the API (Sonnet 5.5) came back with `stop_reason: refusal` and no
output, while ordinary requests work. Rewriting another model's reasoning traces for
training seems to be blocked on the API side. I did not try to reword the request around
it. API spend: about $0.01. `rewrite_traces.py` stays in the repo but is unused.

Replacement, **arm T**: the same removal list applied by rules I wrote (`build_arms.py`),
on top of arm S:

1. Start from the S thinking (cut at the first committed answer, no gold).
2. Delete sentences about the output format (final answer line, example, "42").
3. Delete pure filler and re-check announcements ("Okay.", "Let me think.", "Let me
   check again.", "Hmm.").
4. Delete opening sentences that mostly copy the question (4-gram overlap).
5. Drop sentence-initial filler ("Okay, so", "Well,", "Hmm,") and the articles a/an/the.

The answer section is kept verbatim. Every T trace still goes through the 0.6B fidelity
filter (it must produce the answer from the edited thinking in at least 3 of 4 samples)
before training.

## 7. Arms as built (`reports/arms_build.txt`), and the queue

I read the cut points and edits by hand (many samples per source) and fixed four rule
bugs before freezing the arms:

1. The format-confusion sentence ("like the example says: Final answer: 42") was used as
   the cut point.
2. Splitting on decimals garbled numbers ("1.15 * J" became "15 * J").
3. The article rule deleted the variable or option "A".
4. A sentence like "is 2*4=8" counted as committing to the answer 2. Now the number must
   be the sentence's result.

S's commitment detector now also accepts a math sentence that states the trace's own
final answer as its result ("so 40 + 8 is 48."), not only "answer is 48". That is the
model's own output, not the gold.

Mean thinking tokens per arm (2,398 questions):

| arm | ARC | CSQA | GSM8K |
|---|---:|---:|---:|
| A0 random correct trace | 471 | 460 | 1,384 |
| A shortest correct trace | 343 | 353 | 1,104 |
| **S** cut at first committed answer | 301 | 315 | 363 |
| **T** S + rule-based removals | 244 | 263 | 298 |

In all arms the answer section after `</think>` is the trace's own, verbatim. On GSM8K
that is a long markdown derivation of about 130 tokens. ThinkingCap answers with just the
final line, and that is a possible next arm.

GPU queue (`queue/`, one job at a time):
- 50 Qwen3.8 low effort
- 55 base eval `--samples 2` → `runs/base-s2`
- 60 fidelity filter for S and T
- 70 LoRA SFT on S and T (r=32, 2 epochs, lr 1e-4, `train_sft.py`) → `adapters/`
- 80 eval S and T `--samples 2` → `runs/S-s2`, `runs/T-s2`

Then `compare.py runs/base-s2 runs/X-s2`.

## 8. Is ThinkingCap just "low effort, built in"? No. (`reports/compare_models.md`)

Base Qwen3.8 with `reasoning_effort=low`, on the same 300 questions (thinking / answer
tokens):

| source | Qwen3.8 xhigh | Qwen3.8 low | ThinkingCap |
|---|---:|---:|---:|
| ARC-Easy | 117 / 10 | 206 / 131 | 66 / 6 |
| CommonsenseQA | 570 / 6 | 193 / 102 | 88 / 4 |
| GSM8K | 230 / 60 | 139 / 113 | 105 / 24 |
| Skywork | 2,061 / 207 | 1,203 / 442 | 1,105 / 135 |

Accuracy is about equal across all three (within 1–4 points per source).

Low effort is a *different mode*, not a shorter version of xhigh:
- normal English (articles per 100 words: 11 vs 2), systematic option-by-option
  elimination, Markdown;
- the worked solution moves out of the thinking into the reply. The answer section grows
  5–20×, so total tokens on easy questions go *up*.

ThinkingCap keeps xhigh's terse register, cuts the thinking at least as much as low
effort does, and also strips the reply to the final line. So prompting for low effort
does not reproduce it, and "sample with a brevity instruction, then distil without it"
would not reproduce it either. The changes in section 4 have to be taught directly.

For our method this means:
- the reply after `</think>` counts too, and must not absorb what is cut from the
  thinking;
- arms S/T keep the 0.6B's verbose GSM8K reply (about 130 tokens). A follow-up arm should
  shorten it to a one-line result plus "Final answer: X", as ThinkingCap does.

Queue additions (same LoRA recipe, `--samples 2` eval):
- **U**: T's thinking with the bare "Final answer: X" reply (tests section 8). Filtered
  with T's fidelity result.
- **A**: the shortest correct trace, uncut. The control: does SFT on the model's own short
  samples shorten it without any cut?

## 9. Base eval, fidelity filter, training

**Base Qwen3-0.6B, `--samples 2`** (`runs/base-s2`, 2.6 h):

| benchmark | accuracy | mean tokens |
|---|---:|---:|
| BFCL irrelevance | 82.7% | 206 |
| BFCL multiple | 78.5% | 221 |
| BFCL simple | 81.6% | 195 |
| IFEval | 58.1% | 656 |
| MBPP+ | 51.5% | 2,150 (7.9% truncated) |
| SVAMP | 88.6% | 692 |

**Fidelity filter** (`fidelity_check.py`, at least 3 of 4 answers right from the fixed
thinking):

| arm | ARC | CSQA | GSM8K | total kept |
|---|---:|---:|---:|---:|
| S | 99% | 100% | 91% | 2,327 |
| T | 99% | 99% | 92% | 2,321 |

So the rule-based removals in T cost almost nothing in answerability compared with S.

**Training fix:** the first run hit a GPU out-of-memory error in the cross-entropy over the
152k vocabulary. `train_sft.py` now runs the transformer body once and computes the
lm_head and loss only at completion positions, in checkpointed chunks. 138 optimizer
steps take about 10 minutes per arm. The loss starts at about 0.35, which is low, as
expected for the model's own traces.

## 10. Result: arm S (cut at the first committed answer) vs base, `--samples 2`

| benchmark | tokens saved | accuracy change |
|---|---:|---:|
| SVAMP | **+35.0%** [+31.4, +38.4] | **−2.05 pp** [−3.65, −0.50] |
| IFEval | **−14.3%** [−21.8, −6.9] | −0.09 pp [−2.77, +2.59] |
| MBPP+ | −1.6% [−8.9, +5.4] | +0.26 pp |
| BFCL (3 subsets) | −0.2 to −1.6% (n.s.) | −3.5 to +0.5 pp (n.s.) |
| **average** | **+6.6%** [+3.9, +9.2] | −0.71 pp [−1.80, +0.49] |

Reading it:
- **The cue transfers to arithmetic.** SVAMP thinking drops from 683 to 439 tokens
  (median output 390 → 246) at a small but real accuracy cost of about 2 pp. SVAMP is the
  only benchmark close to the training data (GSM8K).
- **It does not transfer to code, function calling or instruction following.** Their
  thinking length is unchanged.
- **IFEval got longer, and the cause is degenerate loops, not longer reasoning.** The
  median rose only 462 → 484, but truncation at 4,096 tokens rose from 2.9% to 4.3%. On
  the same prompts: 46 truncated for S vs 15 for base. The truncated outputs are drafts
  inside the thinking that degenerate into repeating a phrase, a Gujarati sentence or a
  `"` character. MBPP+ truncation also rose (7.9% → 10.4%).

Likely cause: the SFT data contains only short ARC/CSQA/GSM8K traces. Nothing in it shows
how to finish open-ended or long work, and the LoRA seems to make the model more prone to
repetition outside that domain. A natural fix (not run yet): mix in the 0.6B's own
cleanly-finished traces on general instruction prompts (not the eval benchmarks), so the
adapter learns "stop when committed" without losing "finish open-ended tasks".

Queued follow-up, **arm MS**: arm S plus the 0.6B's own cleanly finished traces on 1,500
general prompts:
- 1,000 from Dolly-15k and 500 from CodeAlpaca-20k (`prepare_general.py`);
- any prompt sharing an 8-gram with an eval question is dropped (9 were);
- 2 samples each, shortest finished one without a repetition loop, used uncut.

MS vs S isolates the effect of the mix on the IFEval/MBPP+ loops and on the transfer of
the stopping cue.

## 11. Result: arm T (S + rule compression, no articles) vs base, `--samples 2`: a failure

| benchmark | tokens saved | accuracy change |
|---|---:|---:|
| SVAMP | +21.0% [+15.7, +26.0] | **−9.00 pp** [−10.95, −7.10] |
| IFEval | **−84.1%** | **−12.38 pp** |
| MBPP+ | **−97.8%** | **−17.20 pp** |
| BFCL multiple | +18.6% | −8.00 pp |
| BFCL simple / irrelevance | +6.3% / −19.8% | n.s. |
| **average** | **−29.6%** [−35.5, −24.1] | **−10.17 pp** [−11.61, −8.76] |

Truncation at the token cap (base → S → T): MBPP+ 7.9% → 10.4% → **44.6%**, IFEval
2.9% → 4.3% → **18.9%**, SVAMP 1.8% → 1.0% → 5.1%. Outputs that finish are short: when T
stops, it stops early. Its tokens go into runaway outputs.

What the runaway outputs look like: the model writes in the article-less register
perfectly (0.49 articles per 100 thinking words, vs 10.75 for base and 11.5 for S), and
then degenerates:
- "Therefore, code is correct. Therefore, function is correct." ×50 (MBPP+);
- "Hambur is sandwich." ×100+ and a Bengali sentence repeated to the cap (IFEval);
- the same three sentences re-reading the question until the cap (SVAMP).

**[Corrected in section 15: V keeps articles and still fails, so articles are only part of
the cause. The reading below is superseded.]**

Reading (superseded): telegraphic English is off-distribution for a 0.6B model. SFT can teach the
register, but generating in it costs the model its ability to stop. ThinkingCap-27B can
afford this register; a 0.6B cannot. The trade is bad on its own terms too: on top of the
other compression rules, removing articles cut only 3–9% of thinking tokens (V vs T
below).

Consequences:
- **U was dropped** (parked in `queue/parked/`). It reuses T's article-less thinking, so it
  would inherit the same failure, and it could not finish before the 12:00 deadline.
- **New arm V = T with articles kept** (`compress(..., articles=True)`). It keeps the
  removals of format talk, filler sentences, copied-question sentences and leading
  filler. Mean thinking tokens:

  | arm | ARC | CSQA | GSM8K |
  |---|---:|---:|---:|
  | S | 301 | 315 | 363 |
  | V | 266 | 288 | 308 |
  | T | 244 | 263 | 298 |

  V uses T's fidelity keep-list: the compression removals are identical, so only the
  articles differ. V vs S shows whether the sentence-level removals are safe on their own.

## 12. Queue state on Oct 10, 07:30 (deadline 12:00)

Running: 82-eval-A (the own-trace control, started 06:54; expected to finish around 10:30).
Then: 84-train-V (~10 min), 90-sample-general → 91-build-MS → 92-train-MS. The two
evals (93-eval-MS, 94-eval-V) will not fit before the deadline; the queue kills them
at 12:00. Both adapters should be ready by then, and each eval takes about 3.5–4 h:

    .venv/bin/python evaluate.py --samples 2 --gpu-memory-utilization 0.8 --adapter adapters/V  --out runs/V-s2
    .venv/bin/python evaluate.py --samples 2 --gpu-memory-utilization 0.8 --adapter adapters/MS --out runs/MS-s2

## 13. Result: arm A (control: own shortest correct trace, uncut) and S vs A

A is trained like S (same questions, LoRA and epochs; 222 vs 138 steps, as its traces are longer) on the
model's own shortest correct trace, with no cut. It separates "fine-tuning on own traces"
from "the stopping cue".

| comparison | tokens saved | accuracy change |
|---|---:|---:|
| A vs base, average | −2.0% [−4.8, +0.8] | +0.04 pp [−1.13, +1.14] |
| A vs base, SVAMP | +4.5% | −0.20 pp |
| **S vs A, average** | **+8.4%** [+5.5, +11.1] | −0.74 pp [−1.87, +0.40] |
| **S vs A, SVAMP** | **+31.9%** [+28.1, +35.7] | −1.85 pp [−3.40, −0.20] |
| S vs A, IFEval | −6.9% [−14.5, 0.0] | +0.74 pp |
| S vs A, MBPP+ / BFCL | n.s. | n.s. |

Truncation at the cap (base / A / S): MBPP+ 7.9% / 11.0% / 10.4%, IFEval 2.9% / 3.9% / 4.3%.

Reading:
- **The extra loops on MBPP+ and IFEval come from the fine-tuning itself, not from the
  cut.** A, with uncut traces, already has them. Picking the shortest of the model's own
  traces alone does almost nothing (−2% tokens).
- **The stopping cue is what saves tokens.** Its effect is real on arithmetic (S vs A:
  −32% tokens on SVAMP, about −2 pp accuracy) and absent outside the training domain.
- So the open problems are (1) transfer of the cue beyond math and multiple choice, and
  (2) the LoRA-induced loops on long open-ended outputs. MS targets the second, and partly
  the first; V tests whether sentence-level compression adds savings safely.

## 14. Final queue state, Oct 10, 12:00 (GPU free)

- adapters/V trained (126 steps, final loss 0.41).
- General sampling: 3,000 samples in 39 min; 1,409 of the 1,500 prompts had a clean finished
  sample. data/arms/MS.jsonl = 2,327 S rows + 1,409 general rows; adapters/MS trained
  (246 steps, final loss 0.31).
- 93-eval-MS was killed at the deadline (exit 124, expected); 94-eval-V never started.
  Neither has results. To run them (about 3.5–4 h each, one at a time), use the two
  commands in section 12, then `compare.py runs/A-s2 runs/X-s2` and
  `compare.py runs/S-s2 runs/X-s2`.

## 15. Results: arms MS and V, `--samples 2`

| arm | vs base: tokens / accuracy | vs A (control) | vs S |
|---|---|---|---|
| **MS** (S + own clean general traces) | **+5.7%** [+3.0, +8.4] / **−0.16 pp** [−1.24, +0.91] | +7.6% / −0.20 pp | −0.9% / +0.54 pp (n.s.) |
| **V** (S + sentence-level compression, articles kept) | **−17.3%** / **−6.02 pp** | −15.0% / −6.06 pp | −25.6% / −5.32 pp |
| T (V + article removal), for reference | −29.6% / −10.17 pp | | |

Truncation at the cap:

| benchmark | base | A | S | MS | V | T |
|---|---:|---:|---:|---:|---:|---:|
| MBPP+ | 7.9% | 11.0% | 10.4% | 9.8% | **27.1%** | **44.6%** |
| IFEval | 2.9% | 3.9% | 4.3% | 4.3% | **17.1%** | **18.9%** |
| SVAMP | 1.8% | 2.2% | 1.0% | 1.3% | 3.7% | 5.1% |

### V: any edit to the model's own text breaks it, not only article removal

V removes only whole sentences (format talk, filler, copies of the question) and leading
filler words, and keeps articles. It still loses 6 pp and loops on IFEval almost as often
as T. Article removal adds damage (MBPP+ truncation 27% → 45%), but it is not the main
cause. Section 11's reading is wrong.

Better explanation: arms A, S and MS train only on text the model itself generated (cut,
but never edited). V and T train on edited text the model would not produce. The training
loss shows the gap: final loss 0.23 (A) and 0.26 (S) vs 0.41 (V) and 0.49 (T). Fitting
off-policy text shifts the 0.6B's distribution more than LoRA SFT on its own samples, and
the shift shows up as loss of control on long, open-ended outputs. Lesson: **for a 0.6B,
compress by selection (which of its own traces, where to cut), not by rewriting.** This also
argues against the planned LLM-rewrite arm (R) for this model size.

### MS: the mix keeps accuracy, does not fix the loops

- MS is the best arm so far: +5.7% tokens saved at −0.16 pp vs base. Accuracy vs S
  recovers slightly everywhere except MBPP+ (+0.54 pp on average, within noise).
- The saving on SVAMP shrinks: MS saves 10% fewer tokens than S there (S: 450 tokens,
  MS: 497, base: 692). The general traces are uncut, which dilutes the stopping cue.
- IFEval thinking is shorter than S (449 vs 531 tokens; +6.4% saved, CI just touches 0).
- Truncation does not improve: IFEval 4.3% (same as S), MBPP+ 9.8% (S 10.4%). The loops
  that LoRA SFT adds (section 13) are not fixed by showing general tasks finishing.

### Where this leaves the method

The only working ingredient is the gold-free stopping cue (cut at the first committed,
unhedged answer), applied by selection to the model's own traces. It saves about 30% on
arithmetic and nothing elsewhere. MBPP+ (2,150 tokens, mostly thinking, and 60% "loop"
by the 12-gram detector even for base) is where the tokens are, and none of the arms
touch it.

Candidate next steps (not started; for review):
1. **Extend the cue to code.** Sample the 0.6B on coding prompts not from MBPP (for
   example CodeAlpaca/MBPP-style from other sources, with tests where available) and cut
   at the first committed code block that passes its own tests or is repeated verbatim
   later. Same selection-only principle.
2. **Fight the base loops directly.** Most MBPP+ tokens are re-checks ("Therefore, code
   is correct" style) and repeated drafts. Train on own traces where the first complete
   draft is followed directly by `</think>`, which is the same cue applied to code.
3. **Smaller LoRA / fewer steps** to reduce the SFT-induced truncation seen in A.

## 16. New base training prompts per domain (for review; nothing sampled or trained)

Built by `prepare_domains.py` into `data/domains/{code,tools,ifeval,math}.jsonl`. Review with
filter counts and one random example per source: `reports/domains_review.md`. Each prompt is
written in its benchmark's own format, and each row keeps its grader data in `meta`. Any prompt
sharing an 8-gram with any of the 2,759 eval prompts is dropped (for IFEval, only the base
request is compared, because the constraint sentences are IFEval's templates by design).

| domain | source | rows | mean prompt tokens (eval) | shape matched to the eval |
|---|---|---:|---:|---|
| code | KodCode-Light-RL-10K (easy/medium, not Package) | 3,000 | 160 (MBPP+ 107) | MBPP+ template: task + one literal `assert` + name reminder; pytest tests kept |
| tools | ToolACE single-turn + xlam-irrelevance-7.5k | 3,000 | 586 (BFCL 414–709) | BFCL's own system prompt; simple 1 function / multiple 2–4 / irrelevance 1–2; 40/30/25% + 5% "missing parameter" declines |
| ifeval | allenai IF_multi_constraints_upto5 | 2,746 | 131 (IFEval 54) | only IFEval's 25 checkable instruction ids; 1/2/3 constraints at 61/28/11% (IFEval 56/33/11) |
| math | GSM8K train + MetaMathQA GSM_SV / GSM_FOBAR | 3,000 | 92 (SVAMP 68) | SVAMP's "Final answer:" instruction; 70% GSM8K, 30% rephrased/backward variants |

Choices worth your review:
- **code:** KodCode prompts are longer specs than MBPP's one-liners, so they are capped at
  300 tokens; that keeps mostly the Filter, Prefill and LeetCode subsets. The example assert
  must use literals only (1,719 dropped whose assert referenced test variables).
- **tools:** ToolACE names with spaces are rewritten to underscores in both the function list
  and the reference call. Its APIs are synthetic web-API style, while BFCL simple is mostly
  maths/science functions.
- **ifeval:** the base requests are English writing/QA tasks from PersonaHub, FLAN, WildChat,
  No Robots and OASST. Dropped: non-English, coding tasks, LaTeX maths, math/safety/table
  sources, more than 3 constraints, and sexual content (128). Only 2,746 rows survive.
- **math:** GSM8K is harder than SVAMP. FOBAR ("if the answer is 4, what is x?") resembles
  SVAMP's question-reversal variants.
- **Not included any more:** ARC and CommonsenseQA. The eval has no multiple-choice task.

## 17. Domain prompts as finally built (supersedes the table in section 16)

Section 16 described a first draft. On review, the training prompts were rebuilt to match
the eval benchmarks more closely (`prepare_domains.py`, `reports/domains_review.md`):

| domain | sources (rows) | matched to |
|---|---|---|
| code | KodCode-V1 Filter 1,326 / Prefill 1,235 / Leetcode 221 / Evol 207 = 2,989 | MBPP+ |
| tools | ToolACE simple 641 / multiple 1,004 / decline 365, xlam irrelevance 976 = 2,986 | BFCL simple / multiple / irrelevance |
| ifeval | IF_multi_constraints_upto5 1/2/3 constraints 963, RLVR-IFeval 1,422 = 2,385 | IFEval |
| math | GSM8K 2,152, MetaMath GSM_SV 614 / GSM_FOBAR 201 = 2,967 | SVAMP |

- **Prompt length is matched to the benchmark,** sampled per decile of the benchmark's
  prompt-token distribution (full chat template), within ±10% of its range. Code prompts are
  imperative KodCode tasks ("Write a function ...") of at most 700 characters, not hard.
- **Every row is gradable with the eval's own graders.** For code, the KodCode reference
  solution must pass its tests in the sandbox. For tools, the gold call must pass BFCL's AST
  checker (ToolACE's "int"/"float" types normalised to BFCL's). For IFEval, only IFEval's
  instruction checkers are used; RLVR-IFeval's constraints are mapped to them, with counts
  capped at IFEval's own maxima.
- **8-gram decontamination against all 2,759 eval prompts,** as before.

**Behaviour check** (`reports/behaviour_check.md`): base Qwen3-0.6B on these prompts vs on
the benchmarks, with the same sampling and caps. Accuracy (train vs eval): code 53.8 vs 51.5,
ifeval 63.7 vs 58.1, math 85.6 vs 88.5, tools 78.9 vs 81.2. Median tokens: 972 vs 1,019,
419 vs 462, 509 vs 390, 186 vs 170. So the training prompts behave like the benchmarks
without being them.

**Sampling** (`behaviour_check.py sample`): 4 samples per prompt at the eval's settings,
45,308 samples in about 25 h. These were graded with the eval graders
(`samples/domains.graded.jsonl`, `reports/domains_samples.md`). 9,337 of the 11,327 prompts
have at least one correct, untruncated sample. On average, the shortest correct sample per
prompt is about 20% shorter than the mean correct one, in every domain.

## 18. Three ways to learn from the same stored samples: SFT, offline GRPO, DPO (`--samples 2`)

All arms are LoRA r=32 (alpha 64, all linear layers). They are trained on the 4 stored
samples per prompt from section 17, token for token, and nothing new is generated.

- **S2, SFT:** train on each prompt's shortest correct sample (9,337 rows; `build_s2.py`).
- **G1, offline GRPO** (`train_grpo_offline.py`):
  - Kimi-k1.5 length reward within each group of 4; the advantage is r − mean (Dr. GRPO);
  - clipped surrogate, with the base model as the old policy; 1 epoch, lr 2e-5;
  - groups with no correct sample are dropped.
- **D, DPO** (`train_dpo_offline.py`):
  - chosen = the shortest correct sample;
  - rejected = the longest wrong or truncated sample, and/or the longest correct sample if
    it is at least 1.5× the chosen;
  - 7,263 pairs; β 0.1; the reference is the base model with the adapter off; 1 epoch,
    lr 2e-5.

| arm | tokens saved | accuracy change |
|---|---|---|
| MS (best earlier arm) | +5.7% [+3.0, +8.4] | −0.16 pp [−1.24, +0.91] |
| S2, SFT | +4.5% [+1.7, +7.2] | +0.90 pp [−0.27, +2.09] |
| G1, offline GRPO | −13.8% [−17.4, −10.4] | −9.41 pp [−10.71, −8.06] |
| **D, DPO** | **+50.2%** [+48.5, +51.9] | **−0.41 pp** [−1.62, +0.80] |

- **SFT on the shortest own sample barely moves length.** The model already produces these
  samples, so imitating them changes little.
- **DPO halves the output at the same accuracy.** The only significant per-benchmark loss is
  SVAMP (−3.85 pp). Thinking is still there, just shorter: SVAMP median thinking falls from
  382 to 160 tokens, and fewer than 0.5% of outputs think for under 20 tokens. Truncation
  falls too (MBPP+ 7.9% → 4.8%). The difference from S2 is the negative sample: learning
  what *not* to write is what moves the length.
- **Offline GRPO fails.** Truncation rises (SVAMP 1.8% → 10.8%, MBPP+ 7.9% → 16.4%), and by
  the end of training 47–49% of tokens are clipped. Reading: with fixed samples and no fresh
  rollouts, the policy drifts far from the data within one epoch. The negative-advantage
  push on long samples then sends probability into behaviour none of the samples contain,
  and that behaviour is looping. Online GRPO might not have this problem; it was not tried.

## 19. What each half of D does: ablations Dw and Dl (`--samples 2`)

Same recipe as D, with one kind of rejected sample only. They have fewer pairs, and
therefore fewer steps, than D, so the effect sizes are not strictly comparable.

| arm | rejected sample | pairs | tokens saved | accuracy change |
|---|---|---:|---|---|
| Dw | wrong or truncated only | 2,931 | +23.0% [+21.0, +25.2] | +0.81 pp [−0.31, +1.95] |
| Dl | longer correct only | 4,332 | +83.6% [+82.7, +84.5] | −13.04 pp [−14.47, −11.44] |
| D | both | 7,263 | +50.2% [+48.5, +51.9] | −0.41 pp [−1.62, +0.80] |

- **Long-correct pairs carry the compression and, alone, collapse the model.** Mean MBPP+
  output falls from 2,150 to 176 tokens, and accuracy drops 7–19 pp on every benchmark. In
  training, Dl separates 100% of pairs with loss 0.0001 while both sides' log-probs fall
  sharply: the pairs are too easy, and DPO pushes everything down.
- **Wrong pairs are an anchor.** On their own they save 23% with no accuracy loss on any
  benchmark: wrong samples are often long or looping, so moving away from them also
  shortens. Mixed with long pairs they hold accuracy.
- **Why SVAMP lost accuracy in D:** math had the most long pairs (1,374) and among the
  fewest wrong pairs (696), because the base model is right 85% of the time on math
  prompts. So math had the weakest anchor.

## 20. Db: balanced pairs, full evaluation (8 samples)

Db = D with, per domain, long pairs capped at the number of wrong pairs (a random subset):
5,862 pairs, with 941 / 724 / 696 / 570 of each kind for code / ifeval / math / tools.
Everything else is as in D. Base and Db were both evaluated with the full recipe, 8 samples
(`runs/base-s8`, `runs/Db-s8`, `reports/compare_Db-s8.txt`).

| benchmark | base acc / tokens | Db acc / tokens | tokens saved | accuracy change |
|---|---|---|---|---|
| BFCL irrelevance | 82.8% / 214 | 88.9% / 115 | +46.1% [+42.8, +49.1] | +6.04 pp [+4.01, +8.13] |
| BFCL multiple | 76.6% / 224 | 75.4% / 130 | +41.9% [+39.3, +44.5] | −1.12 pp [−4.62, +2.56] |
| BFCL simple | 82.2% / 193 | 80.8% / 125 | +35.3% [+33.1, +37.5] | −1.41 pp [−2.94, +0.19] |
| IFEval | 57.8% / 694 | 57.5% / 314 | +54.7% [+52.2, +57.0] | −0.28 pp [−2.10, +1.55] |
| MBPP+ | 52.1% / 2,115 | 50.8% / 1,418 | +33.0% [+29.3, +36.6] | −1.22 pp [−2.98, +0.46] |
| SVAMP | 88.4% / 684 | 86.0% / 298 | +56.4% [+55.0, +57.8] | −2.41 pp [−3.41, −1.47] |
| **average** | | | **+47.0%** [+45.8, +48.1] | **−0.78 pp** [−1.49, −0.05] |

- **This is the first full-recipe result:** 47% fewer tokens at −0.78 pp. ThinkingCap's
  published figures, on its own benchmarks, are 37% fewer reasoning tokens at −0.8 pp.
- **SVAMP still loses about 2.4 pp.** Whether balancing helped cannot be read off directly:
  D was measured with 2 samples and Db with 8. A `--samples 2` eval of Db would settle it.
- **Selection caveat:** Db was designed after seeing D's results on the same benchmark
  questions. The 6 extra samples per question are fresh, but the questions are not, so
  this estimate is slightly optimistic.
- Truncation falls (MBPP+ 7.3% → 3.8%, IFEval 3.5% → 0.6%), and BFCL irrelevance improves
  by 6 pp: the model declines unanswerable calls more readily.

### Trade-off curve (points from this method)

| model | eval | tokens saved | accuracy change |
|---|---|---|---|
| Dw | 2 samples | +23.0% | +0.81 pp |
| Db | **8 samples** | +47.0% | −0.78 pp |
| D | 2 samples | +50.2% | −0.41 pp |
| Dl | 2 samples | +83.6% | −13.04 pp |

### Not run yet (held for review)
1. A `--samples 2` eval of Db, for a direct comparison with D.
2. DPO plus an SFT term on the chosen sample (RPO), as a second fix for SVAMP.
3. A second DPO round on fresh samples from Db (iterative DPO).
