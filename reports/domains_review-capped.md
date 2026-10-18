# Base training prompts per domain (`prepare_domains.py`)

Prompts only: no model output yet. Each row keeps its grader data in `meta`.

| domain | rows built | after filters | written | mean / max prompt tokens |
|---|---:|---:|---:|---:|
| code | 6467 | 3929 | 3000 | 160 / 300 |
| tools | 8169 | 7971 | 3000 | 586 / 1191 |
| ifeval | 3671 | 2746 | 2746 | 131 / 300 |
| math | 87473 | 41580 | 3000 | 92 / 198 |

## code

Dropped:

- prompt over 300 tokens: 2013
- no one-line assert on literals to show: 1719
- hard: 1385
- 8-gram overlap with an eval prompt: 525
- package subset: 429

Example 8-grams that caused a drop: `write a python function to count the number`; `the minimum difference between any two elements in`; `write a function to calculate the sum of`; `write a function to calculate the sum of`; `the index of the first occurrence of a`; `a function to calculate the sum of the`; `10 20 30 40 50 60 70 80`; `the greatest common divisor gcd of two numbers`

Written, by source:

- kodcode/Algorithm: 2
- kodcode/Apps: 204
- kodcode/Code_Contests: 29
- kodcode/Codeforces: 16
- kodcode/Data_Structure: 7
- kodcode/Evol: 206
- kodcode/Filter: 1142
- kodcode/Leetcode: 399
- kodcode/Prefill: 719
- kodcode/Taco: 276

Random examples (one per source):

### kodcode/Algorithm (qid 0, 283 prompt tokens)

**user:**

```text
Write a complete Python solution to the following task. Return the solution in one Python code block.

"""
### Question

You are given a string `s` representing a numeric expression containing digits and the operators `+` and `-`. Your task is to write a function `calculate_expression(s: str) -> int` that evaluates the expression and returns the result as an integer.

#### Constraints:
- The input string can contain spaces, which should be ignored.
- The input string always represents a valid expression.
- The input string will contain at least one digit.

#### Example:

```python
>>> calculate_expression("3+5-2+8")
14
>>> calculate_expression("10 + 22 - 5")
27
>>> calculate_expression("10-2-3")
5
>>> calculate_expression("0 - 1 + 1")
0
>>> calculate_expression("54")
54
```

#### Notes:
- You should handle spaces between numbers and operators gracefully.
- Ensure your solution can handle integer overflow and large inputs efficiently.

Write the function `calculate_expression` to solve the above problem.
assert calculate_expression("3+5-2+8") == 14
"""

The tests call `calculate_expression` and nothing else, so any other function name fails. Define `calculate_expression`, spelled exactly like that.
```

tests (1 test functions), difficulty easy

### kodcode/Apps (qid 123, 121 prompt tokens)

**user:**

```text
Write a complete Python solution to the following task. Return the solution in one Python code block.

"""
Write a function called `flatten_list` which takes a list of lists and returns a single list with all the elements of the nested lists in the same order. The function should only handle a depth of one (i.e., it will not need to handle lists of lists of lists).
assert flatten_list([]) == []
"""

The tests call `flatten_list` and nothing else, so any other function name fails. Define `flatten_list`, spelled exactly like that.
```

tests (5 test functions), difficulty easy

### kodcode/Code_Contests (qid 226, 245 prompt tokens)

**user:**

```text
Write a complete Python solution to the following task. Return the solution in one Python code block.

"""
You are given a string containing only lowercase alphabets. Your task is to find the longest substring without repeating characters. The substring must be a contiguous block of characters within the string.

Write a function `longest_unique_substring` that takes a single string as input and returns the longest substring without repeating characters. If there are multiple substrings with the same maximum length, return the one that appears first.

Input

A single string `s` containing only lowercase alphabets ('a'-'z'). 

Output

A single string which is the longest substring without repeating characters.

Constraints

1 ≤ |s| ≤ 10^5

SAMPLE INPUT
abcabcbb

SAMPLE OUTPUT
abc

SAMPLE INPUT
bbbbb

SAMPLE OUTPUT
b

SAMPLE INPUT
pwwkew

SAMPLE OUTPUT
wke
assert longest_unique_substring("abcabcbb") == "abc"
"""

The tests call `longest_unique_substring` and nothing else, so any other function name fails. Define `longest_unique_substring`, spelled exactly like that.
```

tests (4 test functions), difficulty medium

### kodcode/Codeforces (qid 238, 241 prompt tokens)

**user:**

```text
Write a complete Python solution to the following task. Return the solution in one Python code block.

"""
You are given a list of integers where every element appears twice except for one. Find the element that appears only once without using extra memory for another list. You should aim for a linear runtime complexity.

For example, given the list [4, 3, 2, 4, 1, 3, 2], it should return 1 as it is the only element that appears once.

Input: A list of integers `nums` where `nums.length >= 1` and all numbers except one appear exactly twice.

Output: A single integer which is the element that appears only once.

Write a function `find_single_number(nums: List[int]) -> int` that takes a list of integers and returns the single element that appears only once.
assert find_single_number([4, 3, 2, 4, 1, 3, 2]) == 1
"""

The tests call `find_single_number` and nothing else, so any other function name fails. Define `find_single_number`, spelled exactly like that.
```

tests (5 test functions), difficulty easy

### kodcode/Data_Structure (qid 253, 298 prompt tokens)

**user:**

```text
Write a complete Python solution to the following task. Return the solution in one Python code block.

"""
You are given a list of integers where each integer except one appears exactly twice. Your task is to implement a function `find_unique(nums)` that finds and returns the single number that does not appear in pairs.

### Input
- A list `nums` of integers (1 ≤ len(nums) ≤ 10^6)
- Each integer in `nums` fits in the range of a 32-bit signed integer.

### Output
- Return the single integer that does not appear twice.

### Constraints
- The input list always has exactly one unique element that appears once, while all other elements appear exactly twice.

### Example
```python
>>> find_unique([4, 1, 2, 1, 2])
4
>>> find_unique([2, 2, 3, 3, 4])
4
>>> find_unique([1])
1
```

### Additional Context
This problem can often be found in coding interviews due to its exploitation of bitwise operations for skill assessment on algorithm efficiency. Make sure to handle large inputs and edge cases effectively.
assert find_unique([4, 1, 2, 1, 2]) == 4
"""

The tests call `find_unique` and nothing else, so any other function name fails. Define `find_unique`, spelled exactly like that.
```

tests (5 test functions), difficulty easy

### kodcode/Evol (qid 355, 231 prompt tokens)

**user:**

```text
Write a complete Python solution to the following task. Return the solution in one Python code block.

"""
In Python, write a function that takes a list of integers and returns a list where each integer is replaced by the product of all the integers in the original list except the integer at that position. 

For instance, given the list [1, 2, 3, 4], the function should return [24, 12, 8, 6] because:
- 24 = 2 * 3 * 4 (product of all integers except 1)
- 12 = 1 * 3 * 4 (product of all integers except 2)
- 8 = 1 * 2 * 4 (product of all integers except 3)
- 6 = 1 * 2 * 3 (product of all integers except 4)
assert product_except_self([]) == []
"""

The tests call `product_except_self` and nothing else, so any other function name fails. Define `product_except_self`, spelled exactly like that.
```

tests (5 test functions), difficulty medium

### kodcode/Filter (qid 712, 153 prompt tokens)

**user:**

```text
Write a complete Python solution to the following task. Return the solution in one Python code block.

"""
Write a Python function to reverse a string, but with a twist: if the string is a palindrome (reads the same backward as forward), the function should return the original string in uppercase. If not, return the reversed string in lowercase. Additionally, the function should ignore case and non-alphanumeric characters when checking for a palindrome.
assert reverse_string("A man, a plan, a canal: Panama") == "A MAN, A PLAN, A CANAL: PANAMA"
"""

The tests call `reverse_string` and nothing else, so any other function name fails. Define `reverse_string`, spelled exactly like that.
```

tests (7 test functions), difficulty easy

### kodcode/Leetcode (qid 2951, 173 prompt tokens)

**user:**

```text
Write a complete Python solution to the following task. Return the solution in one Python code block.

"""
You are given two strings, `str1` and `str2`. Your task is to determine if `str1` can be rearranged to form `str2`. In other words, check if the two strings are anagrams of each other. An anagram of a string is the rearrangement of its characters to form another string using all the original characters exactly once. Return `true` if `str1` can be rearranged to form `str2`, and `false` otherwise.
assert are_anagrams("listen", "silent") == True
"""

The tests call `are_anagrams` and nothing else, so any other function name fails. Define `are_anagrams`, spelled exactly like that.
```

tests (4 test functions), difficulty easy

### kodcode/Prefill (qid 2440, 87 prompt tokens)

**user:**

```text
Write a complete Python solution to the following task. Return the solution in one Python code block.

"""
Create a function that takes in an integer and returns the number of its prime factors.
assert count_prime_factors(1) == 0
"""

The tests call `count_prime_factors` and nothing else, so any other function name fails. Define `count_prime_factors`, spelled exactly like that.
```

tests (4 test functions), difficulty easy

### kodcode/Taco (qid 1508, 162 prompt tokens)

**user:**

```text
Write a complete Python solution to the following task. Return the solution in one Python code block.

"""
Implement a function to calculate the factorial of a non-negative integer. Factorial is the product of all positive integers less than or equal to the given number. The factorial of 0 is defined to be 1.

Example:
Given num = 5, return 120.
Given num = 0, return 1.

Do not use any built-in factorial function or libraries.

Credits: Special thanks to @username1 for providing this problem and creating all test cases.
assert factorial(0) == 1
"""

The tests call `factorial` and nothing else, so any other function name fails. Define `factorial`, spelled exactly like that.
```

tests (6 test functions), difficulty easy


## tools

Dropped:

- xlam-irrelevance: no tools or >2 (BFCL irrelevance has 1): 3721
- toolace: unreadable or >4 functions (BFCL has at most 4): 3432
- toolace: not exactly one readable call: 2613
- toolace multi-turn: 800
- 8-gram overlap with an eval prompt: 107
- prompt over 1200 tokens: 77
- toolace: function name not a Python identifier: 33
- explicit content: 32
- duplicate prompt: 14

Example 8-grams that caused a drop: `calculate the probability of rolling a sum of`; `the legend of zelda breath of the wild`; `area and circumference of a circle with a`; `with an annual interest rate of 5 over`; `with an annual interest rate of 5 and`; `the area of a triangle with a base`; `2 3 4 5 6 7 8 9`; `calculate the area of a triangle with base`

Written, by source:

- toolace/decline: 150
- toolace/multiple: 900
- toolace/simple: 1200
- xlam/irrelevance: 750

Random examples (one per source):

### toolace/decline (qid 127, 327 prompt tokens)

**system:**

```text
[BFCL system prompt with 1 function(s): Animate.deleteAnimation]
```

**user:**

```text
I need to delete an animation.
```

reference: `null`  original answer: `The "Animate.deleteAnimation" function can delete an animation, but it lacks the required "animationID" parameter. Please provide the unique ID of the animation you wish to delete.`

### toolace/multiple (qid 551, 708 prompt tokens)

**system:**

```text
[BFCL system prompt with 3 function(s): getResearchPaper, getAirportCode, getRunwayLength]
```

**user:**

```text
I am writing a research paper on airport runway lengths. I need data on the length of runway 4 at John F. Kennedy International Airport. Also, could you fetch research papers related to 'airport safety' published by John D. in 2020 and 'airport efficiency' published by Sarah R. in 2021?
```

reference: `[{"getAirportCode": {"airport_name": ["John F. Kennedy International Airport"]}}]`  original answer: `[getAirportCode(airport_name="John F. Kennedy International Airport")]`

### toolace/simple (qid 1258, 667 prompt tokens)

**system:**

```text
[BFCL system prompt with 1 function(s): roomReservation.create]
```

**user:**

```text
I need to book a room for John Doe, looking for a Suite, check-in on 20231215 at 3:00 PM, and check-out on 20231220 at 11:00 AM. Can you handle this reservation?
```

reference: `[{"roomReservation.create": {"reservationDetails": [{"guestName": "John Doe", "roomType": "Suite", "checkInDate": "20231215 1500", "checkOutDate": "20231220 1100"}]}}]`  original answer: `[roomReservation.create(reservationDetails={"guestName": "John Doe", "roomType": "Suite", "checkInDate": "20231215 1500", "checkOutDate": "20231220 1100"})]`

### xlam/irrelevance (qid 2639, 388 prompt tokens)

**system:**

```text
[BFCL system prompt with 1 function(s): stock_splits]
```

**user:**

```text
Can you fetch the recommendation trend for Ford and the latest quarterly cash flow statement for General Motors?
```

reference: `null`  original answer: ``


## ifeval

Dropped:

- a constraint IFEval cannot check: 77152
- base request from a math/code/safety/table/science set: 7658
- base request is a math problem (LaTeX): 2866
- base request not English, or has <unk> (IFEval asks in English): 2492
- prompt over 300 tokens: 841
- base request is a coding task: 706
- more than 3 constraints (IFEval has 1-3): 700
- sexual content: 128
- 8-gram overlap with an eval prompt: 63
- duplicate prompt: 21

Example 8-grams that caused a drop: `title wrapped in double angular brackets i e`; `put your entire response in double quotation marks`; `your answer must contain one of the following`; `the number of words in all capital letters`; `wrapped in double angular brackets i e title`; `please include the exact phrase in your response`; `in double angular brackets i e title and`; `your answer must contain one of the following`

Written, by source:

- if_multi/1: 1686
- if_multi/2: 772
- if_multi/3: 288

Instruction ids (all rows): keywords:letter_frequency 210, keywords:frequency 202, change_case:english_capital 189, keywords:forbidden_words 184, detectable_format:title 183, length_constraints:number_sentences 183, detectable_content:postscript 179, change_case:capital_word_frequency 177, length_constraints:number_words 177, detectable_format:number_bullet_lists 177, detectable_format:number_highlighted_sections 176, startend:end_checker 170, length_constraints:number_paragraphs 169, change_case:english_lowercase 169, length_constraints:nth_paragraph_first_word 168, punctuation:no_comma 167, startend:quotation 167, keywords:existence 166, detectable_content:number_placeholders 157, detectable_format:multiple_sections 152, language:response_language 144, combination:two_responses 115, detectable_format:json_format 114, combination:repeat_prompt 112, detectable_format:constrained_response 87

Random examples (one per source):

### if_multi/1 (qid 1562, 179 prompt tokens)

**user:**

```text
Create a JSON object that describes the contributions of an administrator to the success of a company. Each sentence should be placed in a separate field named 'sentence_1', 'sentence_2', etc. Use the following description as your source:

The administrator plays a crucial role in the company by ensuring that all operations run smoothly. Their ability to organize and manage daily tasks allows the team to focus on their core responsibilities without unnecessary distractions. They also handle communication between different departments, which facilitates better collaboration and efficiency. By maintaining accurate records and handling administrative tasks diligently, they help the company avoid costly mistakes. Additionally, their support in preparing reports and presentations aids in strategic decision-making processes. The administrator's role in fostering a positive work environment contributes significantly to employee satisfaction and retention. In your response, words with all capital letters should appear less than 36 times.
```

instructions: `['change_case:capital_word_frequency']` kwargs: `[{"capital_frequency": 36, "capital_relation": "less than"}]`

### if_multi/2 (qid 1809, 110 prompt tokens)

**user:**

```text
Write a short song verse that includes three highlighted sections. Use the word "freedom" at least four times and the word "chains" at least twice. The verse should be exactly six sentences long. Entire output should be wrapped in JSON format. You can use markdown ticks such as ```. Your response must have 9 sections. Mark the beginning of each section with SECTION X, such as:
SECTION 1
[content of section 1]
SECTION 2
[content of section 2]
```

instructions: `['detectable_format:json_format', 'detectable_format:multiple_sections']` kwargs: `[{}, {"section_spliter": "SECTION", "num_sections": 9}]`

### if_multi/3 (qid 2574, 195 prompt tokens)

**user:**

```text
You are given a sentence in English. Your job is to translate the English sentence into Spanish.

Input: Consider Input: In the mornings it is terrible, and the afternoon is worst.

Output: Por las mañanas es terrible y por la tarde es aún peor.


Input: Consider Input: For weeks, all the volunteers and locals alike had been finding similar things.

Output: Durante semanas, tanto los voluntarios como los vecinos encontraron cosas similares.


Input: Consider Input: Now, lots of companies fold at that point, but Tim chose to fight.

 Finish your response with this exact phrase That's my answer.. No other words should follow this phrase. Your answer must contain exactly 3 bullet points. Use the markdown bullet points such as:
* This is point 1. 
* This is point 2 Include keywords ['drop', 'effort', 'finding', 'prompt'] in the response.
```

instructions: `['startend:end_checker', 'detectable_format:number_bullet_lists', 'keywords:existence']` kwargs: `[{"end_phrase": "That's my answer."}, {"num_bullets": 3}, {"keywords": ["drop", "effort", "finding", "prompt"]}]`


## math

Dropped:

- duplicate prompt: 45780
- prompt over 200 tokens: 90
- 8-gram overlap with an eval prompt: 23

Example 8-grams that caused a drop: `the same amount of money how much money`; `much do you have to pay to buy`; `pieces of candy how many pieces of candy`; `of gum there are 7 pieces in each`; `each school bus how many buses are needed`; `much do you have to pay to buy`; `of gum there are 7 pieces in each`; `much do you have to pay to buy`

Written, by source:

- gsm8k: 2100
- metamath/GSM_FOBAR: 450
- metamath/GSM_SV: 450

Random examples (one per source):

### gsm8k (qid 402, 66 prompt tokens)

**user:**

```text
Bruce can make 15 batches of pizza dough using a sack of flour. If he uses 5 sacks of flour per day, how many pizza doughs can he make in a week?

The last line of your reply must start with 'Final answer:' followed by the number alone.
```

answer: `525`

### metamath/GSM_FOBAR (qid 2386, 98 prompt tokens)

**user:**

```text
Nick has 60 cans. Each can takes up x square inches of space before being compacted and 20% of that amount after being compacted. How much space do all the cans take up after being compacted?
If we know the answer to the above question is 360, what is the value of unknown variable x?

The last line of your reply must start with 'Final answer:' followed by the number alone.
```

answer: `30`

### metamath/GSM_SV (qid 2756, 63 prompt tokens)

**user:**

```text
The teachers divided the group of students into 3 groups of 8. But x students left early. There remain 22. What is the value of unknown variable x?

The last line of your reply must start with 'Final answer:' followed by the number alone.
```

answer: `2`

