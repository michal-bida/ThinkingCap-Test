# Behaviour check: base Qwen3-0.6B on the training prompts vs on the benchmarks

Training side: `samples/domains.jsonl` (11,327 prompts, 45,308 samples). Benchmark side: `runs/base-s2` (same model, same sampling and caps, 2 samples per prompt).

| set | n | accuracy % | mean tokens | median tokens | mean thinking | truncated % |
|---|---|---:|---:|---:|---:|---:|
| **train code** | 11956 (11956 graded) | 53.8 | 1588 | 972 | 1532 | 3.7 |
| eval mbpp | 756 (756 graded) | 51.5 | 2150 | 1019 | 2105 | 7.9 |
| **train ifeval** | 9540 (9540 graded) | 63.7 | 671 | 419 | 480 | 3.5 |
| eval ifeval | 1082 (1082 graded) | 58.1 | 656 | 462 | 406 | 2.9 |
| **train math** | 11868 (11868 graded) | 85.6 | 786 | 509 | 768 | 2.5 |
| eval svamp | 2000 (2000 graded) | 88.5 | 692 | 390 | 683 | 1.8 |
| **train tools** (None) | 1460 (1460 graded) | 82.4 | 254 | 186 | 227 | 0.0 |
| **train tools** (all) | 11944 (11944 graded) | 78.9 | 247 | 186 | 224 | 0.0 |
| eval bfcl (all) | 1680 (1680 graded) | 81.2 | 204 | 170 | 179 | 0.0 |
| **train tools** (bfcl_irrelevance) | 3904 (3904 graded) | 92.4 | 280 | 222 | 256 | 0.1 |
| eval bfcl_irrelevance | 480 (480 graded) | 82.7 | 206 | 160 | 179 | 0.0 |
| **train tools** (bfcl_multiple) | 4016 (4016 graded) | 68.7 | 254 | 180 | 233 | 0.0 |
| eval bfcl_multiple | 400 (400 graded) | 78.5 | 221 | 192 | 198 | 0.0 |
| **train tools** (bfcl_simple_python) | 2564 (2564 graded) | 72.1 | 181 | 145 | 159 | 0.0 |
| eval bfcl_simple_python | 800 (800 graded) | 81.6 | 195 | 165 | 170 | 0.0 |

Per source (training side only):

| source | n | accuracy % | mean tokens | median tokens | mean thinking | truncated % |
|---|---|---:|---:|---:|---:|---:|
| gsm8k | 8608 (8608 graded) | 85.1 | 770 | 494 | 755 | 2.5 |
| if_multi/1 | 2080 (2080 graded) | 65.6 | 827 | 442 | 603 | 7.0 |
| if_multi/2 | 1376 (1376 graded) | 47.0 | 780 | 483 | 468 | 5.7 |
| if_multi/3 | 396 (396 graded) | 27.3 | 850 | 514 | 427 | 6.1 |
| kodcode/Evol | 828 (828 graded) | 63.6 | 1753 | 1097 | 1690 | 4.3 |
| kodcode/Filter | 5304 (5304 graded) | 50.7 | 1557 | 994 | 1500 | 3.0 |
| kodcode/Leetcode | 884 (884 graded) | 41.2 | 3025 | 1748 | 2954 | 14.1 |
| kodcode/Prefill | 4940 (4940 graded) | 57.8 | 1336 | 850 | 1287 | 2.4 |
| metamath/GSM_FOBAR | 804 (804 graded) | 88.2 | 830 | 544 | 806 | 2.5 |
| metamath/GSM_SV | 2456 (2456 graded) | 86.4 | 828 | 546 | 800 | 2.5 |
| rlvr_ifeval/1 | 5688 (5688 graded) | 69.6 | 575 | 391 | 441 | 1.6 |
| toolace/decline | 1460 (1460 graded) | 82.4 | 254 | 186 | 227 | 0.0 |
| toolace/multiple | 4016 (4016 graded) | 68.7 | 254 | 180 | 233 | 0.0 |
| toolace/simple | 2564 (2564 graded) | 72.1 | 181 | 145 | 159 | 0.0 |
| xlam/irrelevance | 3904 (3904 graded) | 92.4 | 280 | 222 | 256 | 0.1 |

Code verdicts: {'pass': 6432, 'fail': 5024, 'timeout': 43, 'crash': 16}. Rows a grader could not read: 0.

