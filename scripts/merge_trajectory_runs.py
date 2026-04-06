#!/usr/bin/env python3
"""Merge multiple trajectory runs into a single combined folder.

Combines ALL runs into runs/h2-trajectory-all/ — no truncation of attempts.
The analysis script can prune to B=3 internally if needed.

Usage:
    python scripts/merge_trajectory_runs.py <run_dir1> <run_dir2> [<run_dir3> ...]

Example (all four runs at once after the final remaining run finishes):
    python scripts/merge_trajectory_runs.py \\
        runs/h2-trajectory-repair_b_20260328_123657 \\
        runs/h2-trajectory-repair_b-easy_medium_20260329_193826 \\
        runs/h2-trajectory-repair_b-easy_medium_remaining_20260401_103759 \\
        runs/h2-trajectory-repair_b-easy_medium_remaining_YYYYMMDD_HHMMSS

Output: runs/h2-trajectory-all/
  - attempts.jsonl  (all attempt records from all runs)
  - hidden_states/  (all .npz files from all runs)

The script aborts if any task_id appears in more than one run.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

OUTPUT_DIR = Path("runs/h2-trajectory-all")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/merge_trajectory_runs.py <run_dir1> <run_dir2> ...")
        print()
        print("Example:")
        print("  python scripts/merge_trajectory_runs.py \\")
        print("    runs/h2-trajectory-repair_b_20260328_123657 \\")
        print("    runs/h2-trajectory-repair_b-easy_medium_20260329_193826 \\")
        print("    runs/h2-trajectory-repair_b-easy_medium_remaining_20260401_103759 \\")
        print("    runs/h2-trajectory-repair_b-easy_medium_remaining_YYYYMMDD_HHMMSS")
        sys.exit(1)

    run_dirs = [Path(p) for p in sys.argv[1:]]

    # Validate all inputs exist before touching anything
    for d in run_dirs:
        if not d.exists():
            print(f"ERROR: {d} does not exist")
            sys.exit(1)
        if not (d / "attempts.jsonl").exists():
            print(f"ERROR: {d}/attempts.jsonl not found")
            sys.exit(1)

    # Check for task_id collisions across runs.
    # Each task has multiple attempt lines in attempts.jsonl (one per attempt),
    # so we collect unique task_ids per run first, then check cross-run uniqueness.
    print("Checking for task_id collisions across all runs...")
    seen: dict[str, str] = {}  # task_id -> run_dir name
    for d in run_dirs:
        run_ids: set[str] = set()
        with open(d / "attempts.jsonl", encoding="utf-8") as f:
            for line in f:
                run_ids.add(json.loads(line)["task_id"])
        for tid in run_ids:
            if tid in seen:
                print(f"ERROR: task_id collision: {tid}")
                print(f"  First seen in: {seen[tid]}")
                print(f"  Also in: {d.name}")
                sys.exit(1)
            seen[tid] = d.name

    print(f"  OK — {len(seen)} unique tasks across {len(run_dirs)} runs, 0 collisions")
    print()

    # Create output directory
    if OUTPUT_DIR.exists():
        print(f"WARNING: {OUTPUT_DIR} already exists — it will be overwritten.")
        response = input("Continue? [y/N] ").strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(0)
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True)
    (OUTPUT_DIR / "hidden_states").mkdir()

    # Merge all runs
    total_attempts = 0
    total_npz = 0

    for d in run_dirs:
        # Append attempts
        n_attempts = 0
        with open(d / "attempts.jsonl", encoding="utf-8") as fin, \
             open(OUTPUT_DIR / "attempts.jsonl", "a", encoding="utf-8") as fout:
            for line in fin:
                fout.write(line)
                n_attempts += 1

        # Copy hidden_states/*.npz
        hs_src = d / "hidden_states"
        hs_dst = OUTPUT_DIR / "hidden_states"
        n_npz = 0
        if hs_src.exists():
            for npz in sorted(hs_src.glob("*.npz")):
                shutil.copy2(npz, hs_dst / npz.name)
                n_npz += 1
        else:
            print(f"  WARNING: no hidden_states/ in {d.name}")

        print(f"  {d.name}: {n_attempts} attempts, {n_npz} npz files")
        total_attempts += n_attempts
        total_npz += n_npz

    # Final summary
    unique_tasks = len(seen)
    print()
    print(f"Output: {OUTPUT_DIR}")
    print(f"  {unique_tasks} tasks")
    print(f"  {total_attempts} total attempt records")
    print(f"  {total_npz} hidden_state files")


if __name__ == "__main__":
    main()
