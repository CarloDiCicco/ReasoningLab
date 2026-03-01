#!/usr/bin/env python3
"""Download HumanEval+ (via evalplus) and convert to ReasoningLab's JSONL task format.

HumanEval+ is the EvalPlus-corrected version of OpenAI's HumanEval benchmark.
It ships 164 coding problems with assertion-style test code that maps directly
to our executor's exec(candidate_code + test_code) model — zero adapter needed.

The test field contains a check(candidate) function with assertions.  We append
a call to check(entry_point) so the executor can run candidate_code + test_code
as a standalone script.

Usage:
    pip install evalplus
    python scripts/prepare_humaneval_plus.py
    python scripts/prepare_humaneval_plus.py --output data/humaneval_plus.jsonl

Output: data/humaneval_plus.jsonl with 164 TaskRecord-compatible entries.

Requires: pip install evalplus (listed in pyproject.toml [data] extra).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_humaneval_plus() -> dict:
    """Load HumanEval+ dataset via the evalplus package.

    Returns a dict keyed by task_id (e.g. "HumanEval/0"), each value is a dict
    with keys: task_id, prompt, entry_point, canonical_solution, test,
    contract, base_input, atol, plus_input.
    """
    try:
        # EvalPlus exposes HumanEval+ through this function.
        # It returns a dict keyed by task_id.
        from evalplus.data import get_human_eval_plus
    except ImportError as exc:
        raise RuntimeError(
            "evalplus is required to prepare HumanEval+. "
            "Install it with: pip install evalplus"
        ) from exc

    print("Loading HumanEval+ dataset via evalplus ...", flush=True)
    # Example key: "HumanEval/0"
    # Example value fields: prompt, entry_point, test, canonical_solution...
    data = get_human_eval_plus()
    print(f"  Loaded {len(data)} problems.", flush=True)
    return data


def _convert_problem(task_id: str, problem: dict) -> dict:
    """Convert one HumanEval+ problem to ReasoningLab TaskRecord format.

    The prompt wraps the function stub in an instruction for instruct-tuned models.
    _extract_candidate_code() in the policy will extract the code block from the
    model's response.

    The test_code is the EvalPlus check(candidate) function with assertions,
    plus a final call to check(entry_point) so the executor can run it standalone.
    """
    # Instruction-style prompt for chat/instruct models.
    # The function stub (with signature + docstring) is placed inside a code fence.
    prompt = (
        "Complete the following Python function. "
        "Return only one Markdown fenced Python block (```python ... ```), with no extra explanation.\n\n"
        # problem["prompt"] already contains the Python stub with the expected
        # function name/signature (e.g., def has_close_elements(...):).
        f"```python\n{problem['prompt'].rstrip()}\n```"
    )

    # The test field contains:
    #   METADATA = {...}
    #   def check(candidate):
    #       assert candidate(...) == ...
    #       ...
    #
    # We append the call to check(entry_point) so the combined script is:
    #   <candidate_code>        (defines the function)
    #   <test_code>             (defines check + calls it)
    # This is the runtime contract used by execute_candidate(): the generated
    # code is concatenated with test_code and executed as one script.
    test_code = problem["test"] + f"\ncheck({problem['entry_point']})"

    return {
        "task_id": task_id,
        # Model sees this prompt at generation time (task.prompt).
        "prompt": prompt,
        # Verifier executes this test payload (task.test_code).
        "test_code": test_code,
        # Stored for metadata/debugging; tests already enforce the same name.
        "entrypoint": problem["entry_point"],
        # Default per-task timeout used by policies unless overridden by CLI.
        "timeout_s": 8.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare HumanEval+ dataset for ReasoningLab."
    )
    parser.add_argument(
        "--output",
        default="data/humaneval_plus.jsonl",
        help="Output JSONL path (default: data/humaneval_plus.jsonl).",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    # Ensure output directory exists (e.g., data/).
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Download/load raw EvalPlus records.
    data = _load_humaneval_plus()

    # 2) Convert raw records into ReasoningLab TaskRecord-compatible dicts.
    records = [_convert_problem(tid, prob) for tid, prob in data.items()]

    # 3) Write newline-delimited JSON (one task per line), producing JSONL (one JSON object per line).
    #    keep non-ASCII as-is for readability if datasets include them.
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} tasks to {out_path}")


if __name__ == "__main__":
    main()
