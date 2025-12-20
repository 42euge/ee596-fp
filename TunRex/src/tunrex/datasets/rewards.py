"""Reward functions for GRPO training.

These reward functions are used to evaluate model outputs during reinforcement
learning training. They check format compliance and answer correctness.
"""

import re

from tunrex.datasets.config import (
    reasoning_start,
    reasoning_end,
    solution_start,
    solution_end,
)

match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
)


def match_format_exactly(prompts, completions, **kwargs):
    return [
        0 if match_format.search(response) is None else 3.0
        for response in completions
    ]


def match_format_approximately(prompts, completions, **kwargs):
    scores = []

    for completion in completions:
        score = 0
        response = completion
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!
        score += 0.5 if response.count(reasoning_start) == 1 else -0.5
        score += 0.5 if response.find(reasoning_start) == 0 else -0.5
        score += 0.5 if response.count(reasoning_end) == 1 else -0.5
        score += 0.5 if response.count(solution_start) == 1 else -0.5
        score += 0.5 if response.count(solution_end) == 1 else -0.5
        scores.append(score)
    return scores


def check_answer(prompts, completions, answer, **kwargs):
    responses = completions

    extracted_responses = [
        guess.group(1) if r is not None and (guess := match_format.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    assert len(extracted_responses) == len(
        answer
    ), f"{extracted_responses} and {answer} have mismatching length"
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        # Correct answer gets 3 points!
        if guess == true_answer:
            score += 3.0
        # Match if spaces are seen
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess) / float(true_answer)
                if ratio >= 0.9 and ratio <= 1.1:
                    score += 0.5
                elif ratio >= 0.8 and ratio <= 1.2:
                    score += 0.25
                else:
                    score -= 1.0  # Penalize wrong answers
            except:
                score -= 0.5  # Penalize
        scores.append(score)
    return scores


def check_numbers(prompts, completions, answer, **kwargs):
    question = kwargs["question"]
    responses = completions

    extracted_responses = [
        guess.group(1) if (guess := match_numbers.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    print("START ============================")
    print(f"Question:\t{question[0]}")
    print(f"Answer:\t{answer[0]}")
    print(f"Response:\t{responses[0]}")
    print(f"Extracted:\t{extracted_responses[0]}")
    print("END ==============================")
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            guess = float(guess.strip())
            scores.append(1.5 if guess == true_answer else 0.0)
        except:
            scores.append(0)
            continue
    return scores
