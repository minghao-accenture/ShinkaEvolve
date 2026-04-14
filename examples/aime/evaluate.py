"""
Evaluator for AIME agent scaffold evolution.

Loads AIME problems, runs the evolved scaffold on each, checks integer answers.
The scaffold calls a local vLLM server (Qwen3-8B) to solve problems.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from shinka.core import run_shinka_eval
from llm_helper import make_llm_fn


# --- Data loading ---

def _find_aime_data() -> Path:
    """Find AIME data directory. Check env var, then common locations."""
    # Env var override
    data_dir = os.environ.get("AIME_DATA_DIR")
    if data_dir and Path(data_dir).exists():
        return Path(data_dir)

    # Walk up from this file to find the repo root's data/aime/
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "data" / "aime"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "AIME data not found. Set AIME_DATA_DIR or ensure data/aime/ exists."
    )


def load_problems(split: str = "validation", max_problems: int = 30) -> List[Dict]:
    """Load AIME problems.

    Args:
        split: "validation" (90 problems, 2022-2024) or "test" (30, 2025).
        max_problems: Cap on number of problems to load.
    """
    data_dir = _find_aime_data()

    if split == "validation":
        path = data_dir / "aime_validation.jsonl"
    elif split == "test":
        path = data_dir / "aime_2025_test.jsonl"
    else:
        raise ValueError(f"Unknown split: {split}")

    if not path.exists():
        raise FileNotFoundError(f"AIME data file not found: {path}")

    problems = []
    with open(path) as f:
        for line in f:
            problems.append(json.loads(line))
            if len(problems) >= max_problems:
                break

    return problems


# --- Validation and scoring ---

# Number of problems to evaluate on (balance speed vs signal)
NUM_EVAL_PROBLEMS = 30


def validate_aime(
    run_output: List[Tuple[int, int]],
    atol: float = 0.0,
) -> Tuple[bool, Optional[str]]:
    """Validate AIME scaffold output.

    Args:
        run_output: List of (predicted, expected) integer tuples.

    Returns:
        (is_valid, error_message)
    """
    if not run_output:
        return False, "No results returned"

    if not isinstance(run_output, list):
        return False, f"Expected list, got {type(run_output)}"

    for i, item in enumerate(run_output):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            return False, f"Result {i}: expected (predicted, expected) tuple, got {item}"

    # Check that at least some answers are valid integers
    valid_count = sum(1 for pred, _ in run_output if isinstance(pred, int) and 0 <= pred <= 999)
    if valid_count == 0:
        return False, "No valid integer answers (0-999) produced"

    return True, None


def get_aime_kwargs(run_index: int) -> Dict[str, Any]:
    """Provide problems and llm_fn to the scaffold."""
    problems = load_problems("validation", max_problems=NUM_EVAL_PROBLEMS)
    llm_fn = make_llm_fn()
    return {"problems": problems, "llm_fn": llm_fn}


def aggregate_aime_metrics(
    results: List[List[Tuple[int, int]]], results_dir: str
) -> Dict[str, Any]:
    """Aggregate AIME scaffold results.

    Computes accuracy = correct / total as the combined_score.
    """
    if not results:
        return {"combined_score": 0.0, "error": "No results"}

    # Take first run's results (num_runs=1)
    pairs = results[0]
    total = len(pairs)
    correct = sum(1 for pred, exp in pairs if pred == exp)
    accuracy = correct / total if total > 0 else 0.0

    # Save detailed results
    try:
        details_path = os.path.join(results_dir, "aime_details.json")
        details = [
            {"predicted": pred, "expected": exp, "correct": pred == exp}
            for pred, exp in pairs
        ]
        with open(details_path, "w") as f:
            json.dump(details, f, indent=2)
    except Exception:
        pass

    return {
        "combined_score": accuracy,
        "public": {
            "correct_count": correct,
            "total_count": total,
            "accuracy": accuracy,
        },
        "private": {
            "per_problem": [
                {"predicted": pred, "expected": exp}
                for pred, exp in pairs
            ],
        },
    }


def main(program_path: str, results_dir: str):
    """Run AIME evaluation using shinka.eval."""
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    def _aggregator(r):
        return aggregate_aime_metrics(r, results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_aime",
        num_runs=1,
        get_experiment_kwargs=get_aime_kwargs,
        validate_fn=validate_aime,
        aggregate_metrics_fn=_aggregator,
    )

    if correct:
        print("Evaluation completed successfully.")
    else:
        print(f"Evaluation failed: {error_msg}")

    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: <too long>")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for k2, v2 in value.items():
                if isinstance(v2, list) and len(v2) > 5:
                    print(f"    {k2}: [{len(v2)} items]")
                else:
                    print(f"    {k2}: {v2}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIME agent scaffold evaluator")
    parser.add_argument(
        "--program_path", type=str, default="initial.py",
        help="Path to evolved program (must contain 'run_aime')",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Dir to save results (metrics.json, correct.json)",
    )
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
