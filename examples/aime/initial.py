# EVOLVE-BLOCK-START
"""AIME agent scaffold: evolve how we prompt and parse LLM output for math."""

import re


def solve_problem(problem_text: str, llm_fn) -> int:
    """Solve a single AIME problem using the LLM.

    Args:
        problem_text: The AIME problem statement.
        llm_fn: Callable that takes a prompt string and returns a response string.

    Returns:
        Integer answer (0-999).
    """
    prompt = (
        "You are solving an AIME (American Invitational Mathematics Examination) problem.\n"
        "Think step by step, then give your final answer as a single integer (0-999).\n"
        "End your response with: Answer: <number>\n\n"
        f"Problem:\n{problem_text}\n\n"
        "Solution:"
    )
    response = llm_fn(prompt)
    return extract_answer(response)


def extract_answer(response: str) -> int:
    """Extract integer answer from LLM response."""
    # Try common patterns
    patterns = [
        r'(?:answer|Answer|ANSWER)\s*(?:is|:)\s*(\d+)',
        r'\\boxed\{(\d+)\}',
        r'###\s*(\d+)',
    ]
    for pat in patterns:
        matches = re.findall(pat, response)
        if matches:
            try:
                val = int(matches[-1])
                if 0 <= val <= 999:
                    return val
            except ValueError:
                continue

    # Fallback: last number in response
    numbers = re.findall(r'\b(\d+)\b', response)
    if numbers:
        try:
            val = int(numbers[-1])
            if 0 <= val <= 999:
                return val
        except ValueError:
            pass

    return -1  # sentinel for "no answer found"


# EVOLVE-BLOCK-END


def run_aime(problems, llm_fn):
    """Entry point called by evaluator. DO NOT MODIFY.

    Args:
        problems: List of problem dicts with 'problem' and 'answer' keys.
        llm_fn: Callable(prompt_str) -> response_str.

    Returns:
        List of (predicted_answer: int, correct_answer: int) tuples.
    """
    results = []
    for p in problems:
        predicted = solve_problem(p["problem"], llm_fn)
        expected = int(p["answer"])
        results.append((predicted, expected))
    return results
