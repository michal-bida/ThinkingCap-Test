# Behaviour check: base Qwen3-0.6B on the training prompts vs on the benchmarks

Training side: `samples/behaviour.jsonl`, one sample per prompt. Benchmark side: `runs/base-s2` (same model, same sampling and caps, 2 samples per prompt).

| set | n | accuracy % | mean tokens | median tokens | mean thinking | truncated % |
|---|---|---:|---:|---:|---:|---:|
| **train code** | 300 (300 graded) | 53.0 | 1683 | 978 | 1626 | 3.3 |
| eval mbpp | 756 (756 graded) | 51.5 | 2150 | 1019 | 2105 | 7.9 |
| **train ifeval** | 300 (300 graded) | 61.3 | 688 | 448 | 527 | 4.0 |
| eval ifeval | 1082 (1082 graded) | 58.1 | 656 | 462 | 406 | 2.9 |
| **train math** | 300 (300 graded) | 85.7 | 771 | 504 | 757 | 2.7 |
| eval svamp | 2000 (2000 graded) | 88.5 | 692 | 390 | 683 | 1.8 |
| **train tools** (None) | 33 (33 graded) | 72.7 | 251 | 182 | 225 | 0.0 |
| **train tools** (all) | 300 (300 graded) | 79.0 | 252 | 182 | 228 | 0.0 |
| eval bfcl (all) | 1680 (1680 graded) | 81.2 | 204 | 170 | 179 | 0.0 |
| **train tools** (bfcl_irrelevance) | 111 (111 graded) | 92.8 | 278 | 226 | 252 | 0.0 |
| eval bfcl_irrelevance | 480 (480 graded) | 82.7 | 206 | 160 | 179 | 0.0 |
| **train tools** (bfcl_multiple) | 94 (94 graded) | 66.0 | 282 | 170 | 260 | 0.0 |
| eval bfcl_multiple | 400 (400 graded) | 78.5 | 221 | 192 | 198 | 0.0 |
| **train tools** (bfcl_simple_python) | 62 (62 graded) | 77.4 | 159 | 136 | 139 | 0.0 |
| eval bfcl_simple_python | 800 (800 graded) | 81.6 | 195 | 165 | 170 | 0.0 |

Per source (training side only):

| source | n | accuracy % | mean tokens | median tokens | mean thinking | truncated % |
|---|---|---:|---:|---:|---:|---:|
| gsm8k | 228 (228 graded) | 86.8 | 682 | 496 | 672 | 2.2 |
| if_multi/1 | 51 (51 graded) | 54.9 | 782 | 423 | 657 | 7.8 |
| if_multi/2 | 51 (51 graded) | 47.1 | 806 | 551 | 480 | 5.9 |
| if_multi/3 | 7 (7 graded) | 28.6 | 1027 | 443 | 295 | 14.3 |
| kodcode/Evol | 23 (23 graded) | 69.6 | 2529 | 1109 | 2461 | 8.7 |
| kodcode/Filter | 138 (138 graded) | 49.3 | 1568 | 1026 | 1511 | 1.4 |
| kodcode/Leetcode | 18 (18 graded) | 33.3 | 2929 | 1962 | 2838 | 11.1 |
| kodcode/Prefill | 121 (121 graded) | 57.0 | 1469 | 848 | 1419 | 3.3 |
| metamath/GSM_FOBAR | 15 (15 graded) | 73.3 | 1271 | 635 | 1234 | 13.3 |
| metamath/GSM_SV | 57 (57 graded) | 84.2 | 993 | 573 | 972 | 1.8 |
| rlvr_ifeval/1 | 191 (191 graded) | 68.1 | 619 | 419 | 513 | 2.1 |
| toolace/decline | 33 (33 graded) | 72.7 | 251 | 182 | 225 | 0.0 |
| toolace/multiple | 94 (94 graded) | 66.0 | 282 | 170 | 260 | 0.0 |
| toolace/simple | 62 (62 graded) | 77.4 | 159 | 136 | 139 | 0.0 |
| xlam/irrelevance | 111 (111 graded) | 92.8 | 278 | 226 | 252 | 0.0 |

Code verdicts: {'pass': 159, 'fail': 131}. Rows a grader could not read: 0.

