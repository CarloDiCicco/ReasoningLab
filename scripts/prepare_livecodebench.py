#!/usr/bin/env python3
"""Download LiveCodeBench (LeetCode subset) and convert to ReasoningLab's JSONL task format.

LiveCodeBench problems are time-windowed from competitive programming contests
(LeetCode, AtCoder, Codeforces), making them contamination-resistant for models
trained before the contest dates.

We filter to LeetCode-only problems because:
  - LeetCode problems use a function-based format (class Solution + method) that
    maps to assertion-style testing compatible with our executor.
  - AtCoder/Codeforces problems use stdin/stdout format which would require
    executor changes.

The script converts LeetCode input/output test pairs into assertion-style Python
test code so our executor can run them via exec(candidate_code + test_code).

Usage:
    pip install datasets huggingface_hub
    python scripts/prepare_livecodebench.py
    python scripts/prepare_livecodebench.py --output data/lcb_leetcode.jsonl
    python scripts/prepare_livecodebench.py --difficulty easy,medium --after 2024-05-01
    python scripts/prepare_livecodebench.py --max-problems 100

Note: Downloading from HuggingFace may be slow without authentication.
      Set HF_TOKEN env var or run `huggingface-cli login` for faster downloads.

Requires: pip install datasets huggingface_hub (listed in pyproject.toml [data] extra).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import textwrap
from pathlib import Path


# ── Dataset loading ───────────────────────────────────────────────────────────

def _load_lcb_dataset(version_tag: str = "release_v5") -> list[dict]:
    """Load LiveCodeBench code generation dataset from HuggingFace.

    The dataset uses a custom loading script that newer versions of the
    `datasets` library may not support.  We fall back to downloading raw
    JSONL files via `huggingface_hub` if `load_dataset` fails.

    Returns a list of problem dicts with fields:
      question_id, question_title, question_content, platform,
      contest_id, contest_date, starter_code, difficulty,
      public_test_cases, private_test_cases, metadata
    """
    # Map version tags to JSONL filenames in the HuggingFace repo.
    # Releases are cumulative: newer versions add another split file.
    version_files = {
        "release_v1": ["test.jsonl"],
        "release_v2": ["test.jsonl", "test2.jsonl"],
        "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
        "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
        "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
        "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
    }

    filenames = version_files.get(version_tag)
    if filenames is None:
        raise ValueError(
            f"Unknown version_tag: {version_tag}. "
            f"Supported: {list(version_files.keys())}"
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download LiveCodeBench. "
            "Install with: pip install huggingface_hub"
        ) from exc

    problems: list[dict] = []
    for filename in filenames:
        print(f"  Downloading {filename} ...", flush=True)
        # hf_hub_download stores files in the local HF cache and returns
        # the local path. Re-running typically reuses cache if unchanged.
        path = hf_hub_download(
            "livecodebench/code_generation_lite",
            filename=filename,
            repo_type="dataset",
        )
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    problems.append(json.loads(line))

    return problems


# ── Conversion helpers ────────────────────────────────────────────────────────

def _parse_starter_code(starter_code: str) -> tuple[str, str] | None:
    """Extract (class_name, method_name) from LeetCode starter_code.

    LeetCode starter code typically looks like:
        class Solution:
            def methodName(self, param1: Type, ...) -> ReturnType:

    Returns None if the pattern is not recognized.
    """
    # Match "class X:" and "def method(self, ...)".
    # We only support instance-method style LeetCode stubs here.
    class_match = re.search(r"class\s+(\w+)", starter_code)
    method_match = re.search(r"def\s+(\w+)\s*\(\s*self", starter_code)
    if class_match and method_match:
        return class_match.group(1), method_match.group(1)
    return None


def _parse_test_cases(test_cases_json: str) -> list[dict]:
    """Parse test case JSON string into a list of {input, output} dicts.

    LCB stores test cases as a JSON string.  The format varies:
      - List of dicts: [{"input": "...", "output": "..."}, ...]
      - Single dict:   {"input": "...", "output": "..."}
    """
    # Some rows have missing/empty test payloads.
    if not test_cases_json or test_cases_json.strip() == "":
        return []
    try:
        parsed = json.loads(test_cases_json)
    except json.JSONDecodeError:
        return []

    # Normalize both accepted shapes to a list for downstream iteration.
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return []


def _build_assertion(method_call: str, expected_output: str) -> str:
    """Build an assertion line, handling common edge cases.

    Returns a Python assertion string like:
        assert sol.method(args) == expected
    """
    # Try to parse expected output as a Python literal.
    try:
        expected = ast.literal_eval(expected_output.strip())
    except (ValueError, SyntaxError):
        # If it is not a Python literal, fall back to raw string expectation.
        expected = expected_output.strip()

    # LCB stores boolean outputs as JSON strings "true"/"false".
    # Convert them to Python booleans so assertions work correctly.
    if expected == "true":
        expected = True
    elif expected == "false":
        expected = False

    return f"assert {method_call} == {repr(expected)}"


def _convert_problem(problem: dict) -> dict | None:
    """Convert one LCB LeetCode problem to ReasoningLab TaskRecord format.

    Returns None if the problem cannot be converted (e.g., unrecognized
    starter_code format or no test cases).
    """
    starter_code = problem.get("starter_code", "")
    parsed = _parse_starter_code(starter_code)
    if parsed is None:
        return None

    class_name, method_name = parsed

    # Build prompt: natural-language statement + exact starter stub.
    # Keeping starter code verbatim preserves the expected class/method signature.
    question_content = problem.get("question_content", "")
    prompt = (
        "Solve the following problem. Complete the Solution class method. "
        "Return only one Markdown fenced Python block (```python ... ```), with no extra explanation.\n\n"
        f"{question_content}\n\n"
        f"```python\n{starter_code.rstrip()}\n```"
    )

    # Parse test cases from both public and private pools.
    # We merge them to increase evaluation coverage.
    public_tests = _parse_test_cases(problem.get("public_test_cases", ""))
    private_tests = _parse_test_cases(problem.get("private_test_cases", ""))
    all_tests = public_tests + private_tests

    if not all_tests:
        return None

    # Build assertion-style test_code.
    # LeetCode test inputs are typically formatted as one value per line,
    # matching the method's parameter order.
    assertions: list[str] = []
    for tc in all_tests:
        tc_input = tc.get("input", "")
        tc_output = tc.get("output", tc.get("expected_output", ""))

        # LeetCode input format: each parameter on a separate line.
        # e.g., "[2,7,11,15]\n9" for twoSum(nums, target).
        input_lines = [line.strip() for line in tc_input.strip().split("\n") if line.strip()]

        # Parse each input line as a Python literal (list/int/str/bool/...).
        # If parsing fails, skip the test case rather than failing the whole task.
        try:
            parsed_args = [ast.literal_eval(line) for line in input_lines]
        except (ValueError, SyntaxError):
            # Skip test cases with unparseable inputs.
            continue

        args_str = ", ".join(repr(a) for a in parsed_args)
        method_call = f"sol.{method_name}({args_str})"

        try:
            assertion = _build_assertion(method_call, str(tc_output))
            assertions.append(assertion)
        except Exception:
            continue

    if not assertions:
        return None

    # Build test code executed by the verifier:
    # 1) instantiate Solution-like class
    # 2) run all generated assertions
    test_lines = [f"sol = {class_name}()"]
    test_lines.extend(assertions)
    test_code = "\n".join(test_lines)

    question_id = problem.get("question_id", problem.get("question_title", "unknown"))

    return {
        "task_id": f"LCB/{question_id}",
        "prompt": prompt,
        "test_code": test_code,
        "entrypoint": method_name,
        "timeout_s": 10.0,  # LCB problems can be more complex → slightly longer timeout
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare LiveCodeBench (LeetCode subset) for ReasoningLab."
    )
    parser.add_argument(
        "--output", default="data/lcb_leetcode.jsonl",
        help="Output JSONL path (default: data/lcb_leetcode.jsonl).",
    )
    parser.add_argument(
        "--version", default="release_v6",
        help="LCB version tag (default: release_v6).",
    )
    parser.add_argument(
        "--difficulty", default="easy,medium",
        help="Comma-separated difficulty filter (default: easy,medium).",
    )
    parser.add_argument(
        "--after", default=None,
        help="Only include problems after this date (YYYY-MM-DD). "
             "Use 2024-05-01 for Qwen3/Qwen3.5 contamination safety "
             "(release_v6 only covers up to ~Apr 2025, so a later cutoff is not feasible).",
    )
    parser.add_argument(
        "--max-problems", type=int, default=None,
        help="Maximum number of problems to include (default: all).",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize CLI difficulty list (e.g., "easy,medium").
    allowed_difficulties = {d.strip().lower() for d in args.difficulty.split(",")}

    print(f"Loading LiveCodeBench ({args.version}) ...", flush=True)
    problems = _load_lcb_dataset(args.version)
    print(f"  Total problems: {len(problems)}", flush=True)

    # Keep only LeetCode-style function/class tasks.
    # Other platforms often require stdin/stdout judge adapters.
    leetcode = [p for p in problems if "leetcode" in str(p.get("platform", "")).lower()]
    print(f"  LeetCode problems: {len(leetcode)}", flush=True)

    # Filter by difficulty.
    filtered = [
        p for p in leetcode
        if str(p.get("difficulty", "")).strip().lower() in allowed_difficulties
    ]
    print(f"  After difficulty filter ({args.difficulty}): {len(filtered)}", flush=True)

    # Optional contamination-control filter.
    # contest_date is expected in ISO format, so string comparison works.
    if args.after:
        filtered = [
            p for p in filtered
            if str(p.get("contest_date", "")) >= args.after
        ]
        print(f"  After date filter (>{args.after}): {len(filtered)}", flush=True)

    # Convert raw dataset rows to ReasoningLab task records.
    records: list[dict] = []
    skipped = 0
    for p in filtered:
        record = _convert_problem(p)
        if record is not None:
            records.append(record)
        else:
            skipped += 1

    if skipped > 0:
        print(f"  Skipped {skipped} problems (unparseable format).", flush=True)

    # Apply max-problems cap after conversion.
    if args.max_problems is not None and len(records) > args.max_problems:
        records = records[: args.max_problems]
        print(f"  Capped to {args.max_problems} problems.", flush=True)

    # Write output.
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} tasks to {out_path}")


if __name__ == "__main__":
    main()
