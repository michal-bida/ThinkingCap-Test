# Base training prompts per domain (`prepare_domains.py`)

Prompts only: no model output yet. Each row keeps its grader data in `meta`.

| domain | rows built | after filters | written | mean / max prompt tokens |
|---|---:|---:|---:|---:|
| code | 35048 | 31002 | 2989 | 110 / 249 |
| tools | 8169 | 6561 | 2986 | 522 / 1279 |
| ifeval | 11186 | 5688 | 2385 | 68 / 399 |
| math | 87473 | 29568 | 2967 | 70 / 110 |

## code

Dropped:

- not an instruction to write code (question, chat or def stub): 57706
- question over 700 characters: 27851
- hard: 27527
- no one-line assert on literals to show: 10161
- not exactly one function: 9377
- 8-gram overlap with an eval prompt: 3817
- prompt length outside the benchmark's range (+-10%): 224
- reference solution does not pass its tests here (unsupported), after selection: 9
- duplicate prompt: 5
- reference solution does not pass its tests here (fail), after selection: 2

Example 8-grams that caused a drop: `write a function to find the length of`; `function to find the element that appears only`; `write a function that takes in a list`; `find the second smallest number in a list`; `2 3 4 5 6 7 8 9`; `write a function to find the area of`; `write a function to find the number of`; `write a function to count the number of`

Prompt tokens, p10 / p25 / p50 / p75 / p90 (written rows vs their benchmark):

| group | n | written | benchmark |
|---|---:|---|---|
| kodcode/Evol vs mbpp | 207 | 102 / 109 / 121 / 140 / 180 | 84 / 90 / 100 / 116 / 142 |
| kodcode/Filter vs mbpp | 1326 | 84 / 90 / 100 / 119 / 143 | 84 / 90 / 100 / 116 / 142 |
| kodcode/Leetcode vs mbpp | 221 | 107 / 112 / 125 / 151 / 188 | 84 / 90 / 100 / 116 / 142 |
| kodcode/Prefill vs mbpp | 1235 | 83 / 89 / 99 / 115 / 134 | 84 / 90 / 100 / 116 / 142 |

Written, by source:

- kodcode/Evol: 207
- kodcode/Filter: 1326
- kodcode/Leetcode: 221
- kodcode/Prefill: 1235

Random examples (one per source):

### kodcode/Evol (qid 19, 101 prompt tokens)

**user:**

```text
Write a complete Python solution to the following task. Return the solution in one Python code block.

"""
Write a function that takes a string as input and returns the string reversed. For example, if the input is "hello", the function should return "olleh".
assert reverse_string("hello") == "olleh"
"""

The tests call `reverse_string` and nothing else, so any other function name fails. Define `reverse_string`, spelled exactly like that.
```

tests (7 test functions), difficulty easy

### kodcode/Filter (qid 362, 87 prompt tokens)

**user:**

```text
Write a complete Python solution to the following task. Return the solution in one Python code block.

"""
Given a string s, reverse the characters of the string without using any built-in string function.
assert reverse_string("hello") == "olleh"
"""

The tests call `reverse_string` and nothing else, so any other function name fails. Define `reverse_string`, spelled exactly like that.
```

tests (7 test functions), difficulty easy

### kodcode/Leetcode (qid 1541, 140 prompt tokens)

**user:**

```text
Write a complete Python solution to the following task. Return the solution in one Python code block.

"""
You are given a list of `n` integers `nums` which could contain both positive and negative numbers. You need to determine if there exists a subarray whose sum is equal to zero. A subarray is any contiguous part of the array. Return `true` if such a subarray exists, otherwise return `false`.
assert subarray_sum_zero_exists([]) == False
"""

The tests call `subarray_sum_zero_exists` and nothing else, so any other function name fails. Define `subarray_sum_zero_exists`, spelled exactly like that.
```

tests (12 test functions), difficulty easy

### kodcode/Prefill (qid 2049, 104 prompt tokens)

**user:**

```text
Write a complete Python solution to the following task. Return the solution in one Python code block.

"""
Create a function that takes a string of text and adds an exclamation mark at the end if it doesn't already end with one. If the text is empty, return an empty string.
assert add_exclamation("") == ""
"""

The tests call `add_exclamation` and nothing else, so any other function name fails. Define `add_exclamation`, spelled exactly like that.
```

tests (3 test functions), difficulty easy


## tools

Dropped:

- xlam-irrelevance: no tools or >2 (BFCL irrelevance has 1): 3721
- toolace: unreadable or >4 functions (BFCL has at most 4): 3432
- toolace: not exactly one readable call: 2613
- prompt length outside the benchmark's range (+-10%): 1012
- toolace multi-turn: 800
- its own gold call fails BFCL's checker (types, nesting): 475
- 8-gram overlap with an eval prompt: 107
- toolace: function name not a Python identifier: 33
- explicit content: 32
- duplicate prompt: 14

Example 8-grams that caused a drop: `calculate the probability of rolling a sum of`; `the legend of zelda breath of the wild`; `the area and circumference of a circle with`; `with an annual interest rate of 5 over`; `with an annual interest rate of 5 and`; `area of a triangle with a base of`; `2 3 4 5 6 7 8 9`; `calculate the area of a triangle with base`

Prompt tokens, p10 / p25 / p50 / p75 / p90 (written rows vs their benchmark):

| group | n | written | benchmark |
|---|---:|---|---|
| toolace/decline vs bfcl_simple_python | 365 | 377 / 395 / 427 / 461 / 508 | 378 / 402 / 420 / 450 / 480 |
| toolace/multiple vs bfcl_multiple | 1004 | 552 / 608 / 693 / 784 / 913 | 558 / 596 / 690 / 777 / 919 |
| toolace/simple vs bfcl_simple_python | 641 | 347 / 381 / 422 / 466 / 522 | 378 / 402 / 420 / 450 / 480 |
| xlam/irrelevance vs bfcl_irrelevance | 976 | 368 / 387 / 412 / 439 / 485 | 378 / 396 / 412 / 431 / 453 |

Written, by source:

- toolace/decline: 365
- toolace/multiple: 1004
- toolace/simple: 641
- xlam/irrelevance: 976

Random examples (one per source):

### toolace/decline (qid 142, 583 prompt tokens)

**system:**

```text
[BFCL system prompt with 2 function(s): Financial_Auto_Complete, CurrencyConversion]
```

**user:**

```text
Can you retrieve the details of two Litecoin transactions? The hashes are '0x9d3a9d0dad3eff8c' and '0x2f5a2dfd8a3e7cb9'.
```

reference: `null`  original answer: `The given functions do not provide the capability to retrieve details of Litecoin transactions based on their hashes. Thus, I cannot fulfill this request with the available functions.`

### toolace/multiple (qid 634, 700 prompt tokens)

**system:**

```text
[BFCL system prompt with 4 function(s): showerthoughts, Get_Poets, Get_Chapter_Information, Filmarks_API]
```

**user:**

```text
I'd love to hear one of those unpredictable food-for-thought ideas that people come up with in the shower. Could you share a random one, please?
```

reference: `[{"showerthoughts": {}}]`  original answer: `[showerthoughts()]`

### toolace/simple (qid 1616, 560 prompt tokens)

**system:**

```text
[BFCL system prompt with 1 function(s): airQualityIndex]
```

**user:**

```text
Can you provide the current air quality index and the main pollutant for Paris, France? I'm particularly concerned about NO2 levels.
```

reference: `[{"airQualityIndex": {"regionCode": ["FR"], "pollutants": [["NO2"]], "time": [""]}}]`  original answer: `[airQualityIndex(regionCode="FR", pollutants=["NO2"])]`

### xlam/irrelevance (qid 1685, 374 prompt tokens)

**system:**

```text
[BFCL system prompt with 1 function(s): bullish_harami]
```

**user:**

```text
What are the SWIFT codes for France and Germany? Also, what is the word equivalent of the number 1,000,000 in English and Spanish?
```

reference: `null`  original answer: ``


## ifeval

Dropped:

- a constraint IFEval cannot check: 77152
- base request from a math/code/safety/table/science set: 7658
- base request not English, or has <unk> (IFEval asks in English): 3318
- base request is a math problem (LaTeX): 3199
- rlvr: count above any in IFEval (e.g. 30 paragraphs): 2877
- rlvr: constraint without an IFEval equivalent: 2486
- duplicate prompt: 2206
- prompt length outside the benchmark's range (+-10%): 1827
- 8-gram overlap with an eval prompt: 1465
- base request is a coding task: 869
- more than 3 constraints (IFEval has 1-3): 700
- rlvr: 'around N' (no IFEval checker): 614
- sexual content: 287

Example 8-grams that caused a drop: `title wrapped in double angular brackets i e`; `put your entire response in double quotation marks`; `must contain one of the following exact phrases`; `the number of words in all capital letters`; `in double angular brackets i e title and`; `please include the exact phrase in your response`; `in double angular brackets i e title and`; `must contain one of the following exact phrases`

Prompt tokens, p10 / p25 / p50 / p75 / p90 (written rows vs their benchmark):

| group | n | written | benchmark |
|---|---:|---|---|
| if_multi/1 vs ifeval/1 | 520 | 34 / 41 / 53 / 68 / 178 | 30 / 35 / 43 / 54 / 71 |
| if_multi/2 vs ifeval/2 | 344 | 53 / 64 / 81 / 101 / 116 | 43 / 48 / 57 / 69 / 82 |
| if_multi/3 vs ifeval/3 | 99 | 70 / 76 / 94 / 105 / 110 | 49 / 55 / 70 / 83 / 90 |
| rlvr_ifeval/1 vs ifeval/1 | 1422 | 32 / 38 / 45 / 55 / 78 | 30 / 35 / 43 / 54 / 71 |

Written, by source:

- if_multi/1: 520
- if_multi/2: 344
- if_multi/3: 99
- rlvr_ifeval/1: 1422

Instruction ids (all rows): keywords:forbidden_words 271, keywords:existence 266, startend:end_checker 240, length_constraints:number_sentences 213, length_constraints:number_words 204, change_case:english_capital 172, punctuation:no_comma 157, startend:quotation 147, detectable_format:title 127, change_case:english_lowercase 125, detectable_format:json_format 114, length_constraints:number_paragraphs 113, combination:two_responses 90, detectable_format:number_bullet_lists 85, keywords:letter_frequency 82, keywords:frequency 80, detectable_format:number_highlighted_sections 73, length_constraints:nth_paragraph_first_word 67, language:response_language 66, detectable_content:postscript 66, change_case:capital_word_frequency 56, detectable_content:number_placeholders 46, detectable_format:multiple_sections 29, combination:repeat_prompt 19, detectable_format:constrained_response 19

Random examples (one per source):

### if_multi/1 (qid 1312, 58 prompt tokens)

**user:**

```text
(In the school literature clubroom...)

Monika: "...it's been 10 minutes. MC still isn't here - where IS he?!" Highlight at least 1 sections in your answer with markdown, i.e. *highlighted section*.
```

instructions: `['detectable_format:number_highlighted_sections']` kwargs: `[{"num_highlights": 1}]`

### if_multi/2 (qid 1614, 55 prompt tokens)

**user:**

```text
Come up with a question and stream of consciousness reasoning that would justify this answer: it is not possible to tell Your entire response should be in English, and in all capital letters. Your response should contain at least 30 sentences.
```

instructions: `['change_case:english_capital', 'length_constraints:number_sentences']` kwargs: `[{}, {"num_sentences": 30, "relation": "at least"}]`

### if_multi/3 (qid 1872, 79 prompt tokens)

**user:**

```text
generate more comprehensive details and examples on, 4. Crafting Compelling Product Descriptions, minimalist tone In your response, the letter k should appear at least 18 times. Highlight at least 2 sections in your answer with markdown, i.e. *highlighted section*. In your response, the word requirement should appear less than 3 times.
```

instructions: `['keywords:letter_frequency', 'detectable_format:number_highlighted_sections', 'keywords:frequency']` kwargs: `[{"letter": "k", "let_frequency": 18, "let_relation": "at least"}, {"num_highlights": 2}, {"keyword": "requirement", "frequency": 3, "relation": "less than"}]`

### rlvr_ifeval/1 (qid 1375, 59 prompt tokens)

**user:**

```text
A young girl in a polka dot dress plays on a beach is a synonym of a girl plays at the beach.
This justifies what answer for what question? Q & A:  In your entire response, refrain from the use of any commas.
```

instructions: `['punctuation:no_comma']` kwargs: `[{}]`


## math

Dropped:

- duplicate prompt: 45780
- prompt length outside the benchmark's range (+-10%): 12102
- 8-gram overlap with an eval prompt: 23

Example 8-grams that caused a drop: `the same amount of money how much money`; `much do you have to pay to buy`; `pieces of candy how many pieces of candy`; `of gum there are 7 pieces in each`; `on each school bus how many buses are`; `much do you have to pay to buy`; `of gum there are 7 pieces in each`; `much do you have to pay to buy`

Prompt tokens, p10 / p25 / p50 / p75 / p90 (written rows vs their benchmark):

| group | n | written | benchmark |
|---|---:|---|---|
| gsm8k vs svamp | 2152 | 57 / 62 / 67 / 74 / 81 | 57 / 62 / 67 / 74 / 82 |
| metamath/GSM_FOBAR vs svamp | 201 | 70 / 73 / 78 / 87 / 99 | 57 / 62 / 67 / 74 / 82 |
| metamath/GSM_SV vs svamp | 614 | 60 / 63 / 67 / 75 / 81 | 57 / 62 / 67 / 74 / 82 |

Written, by source:

- gsm8k: 2152
- metamath/GSM_FOBAR: 201
- metamath/GSM_SV: 614

Random examples (one per source):

### gsm8k (qid 2949, 90 prompt tokens)

**user:**

```text
My mother celebrated her birthday with a total of 60 guests. Half of the guests are women, 15 are men, and the rest are children. In the middle of the celebration, 1/3 of the men and 5 children left. How many people stayed and enjoyed the birthday celebration?

The last line of your reply must start with 'Final answer:' followed by the number alone.
```

answer: `50`

### metamath/GSM_FOBAR (qid 2248, 82 prompt tokens)

**user:**

```text
Tilly needs to sell 100 bags at $x per bag to make $300 in profit. How much did she buy each bag for?
If we know the answer to the above question is 7, what is the value of unknown variable x?

The last line of your reply must start with 'Final answer:' followed by the number alone.
```

answer: `10`

### metamath/GSM_SV (qid 2288, 58 prompt tokens)

**user:**

```text
Madeline has $48 and her brother has x% as much. They have 72 together. What is the value of unknown variable x?

The last line of your reply must start with 'Final answer:' followed by the number alone.
```

answer: `50`

