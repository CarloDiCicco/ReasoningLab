#!/usr/bin/env python3
"""Phase B — bug isolation.

Grade a run's SAVED generated code under the OLD buggy (public-only) test suite,
holding the generations fixed and varying only the test suite. This isolates the
effect of the LiveCodeBench private-test fix from run-to-run generation
stochasticity: the hidden states are unchanged, only the pass/fail LABELS change
(now derived from the buggy public-only tests instead of the full corrected ones).

Reads:  <RUN>/attempts.jsonl  (must contain the `code` field — code-saving on)
        data/lcb_all.jsonl     (buggy public-only test suites)
Writes: <OUT>/attempts.jsonl   (same schema, `passed`/`failure_type` re-derived
        under buggy tests; `code` preserved) so analyze_trajectories.py can run
        on it directly with the ORIGINAL hidden_states symlinked/pointed in.

Usage:
    python scripts/regrade_under_buggy_tests.py \
        --run <RUN_DIR_WITH_SAVED_CODE> \
        --tests data/lcb_all.jsonl \
        --out  <OUTPUT_RUN_DIR> \
        --timeout 8.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reasoninglab.verify.executor import execute_candidate
from reasoninglab.verify.taxonomy import classify_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path, help="run dir with attempts.jsonl (code saved)")
    ap.add_argument("--tests", required=True, type=Path, help="test-suite jsonl to grade against")
    ap.add_argument("--out", required=True, type=Path, help="output run dir")
    ap.add_argument("--timeout", type=float, default=8.0, help="verifier timeout (paper buggy run used 8.0)")
    args = ap.parse_args()

    # Load the test suites to grade against, keyed by task_id.
    tests: dict[str, str] = {}
    with args.tests.open() as f:
        for line in f:
            r = json.loads(line)
            tests[r["task_id"]] = r["test_code"]
    print(f"Loaded {len(tests)} test suites from {args.tests}")

    attempts_path = args.run / "attempts.jsonl"
    records = [json.loads(l) for l in attempts_path.open()]
    with_code = sum(1 for r in records if r.get("code"))
    print(f"Loaded {len(records)} attempts ({with_code} with saved code) from {attempts_path}")
    if with_code == 0:
        raise SystemExit("No saved code in this run — cannot re-grade. Re-run with code-saving on.")

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "attempts.jsonl"

    flips = 0
    n_graded = 0
    with out_path.open("w", encoding="utf-8") as out:
        for r in records:
            code = r.get("code")
            tid = r["task_id"]
            if code is None or tid not in tests:
                # Can't regrade — keep as-is (shouldn't happen for repair_b runs).
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                continue
            execution = execute_candidate(
                candidate_code=code,
                test_code=tests[tid],
                timeout_s=args.timeout,
            )
            ft = classify_result(execution)
            new_passed = bool(execution.passed)
            if new_passed != r["passed"]:
                flips += 1
            r = dict(r)
            r["passed"] = new_passed
            r["failure_type"] = ft.value if hasattr(ft, "value") else str(ft)
            r["stderr"] = (execution.stderr or "")[:4000] if execution.stderr else None
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_graded += 1
            if n_graded % 200 == 0:
                print(f"  graded {n_graded}/{len(records)}  (flips so far: {flips})")

    print(f"\nDone. Re-graded {n_graded} attempts under {args.tests.name}.")
    print(f"Label flips vs original run: {flips}")
    print(f"Wrote {out_path}")
    print(f"\nNOTE: point analyze_trajectories at this dir but with the ORIGINAL")
    print(f"hidden_states (they are unchanged). E.g. symlink:")
    print(f"  ln -s $(realpath {args.run}/hidden_states) {args.out}/hidden_states")


if __name__ == "__main__":
    main()
