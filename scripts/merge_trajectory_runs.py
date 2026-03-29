#!/usr/bin/env python3
"""Merge a new trajectory run into the h2-trajectory-merged folder.

After running the easy+medium Repair-B trajectory experiment, use this script
to append its attempts.jsonl and hidden_states/*.npz into the merged folder
that already contains the filtered hard task data (attempts 0-2 only).

Usage:
    python scripts/merge_trajectory_runs.py runs/h2-trajectory-repair_b-easy_medium_YYYYMMDD_HHMMSS

The run directory name (with timestamp) is passed as a command-line argument.
You provide the actual name after the run completes.

The script will:
  1. Append attempts.jsonl lines to h2-trajectory-merged/attempts.jsonl
  2. Copy hidden_states/*.npz files to h2-trajectory-merged/hidden_states/
  3. Verify no task_id collisions (same task in both runs)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

MERGED_DIR = Path("runs/h2-trajectory-merged")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/merge_trajectory_runs.py <new_run_dir>")
        print("Example: python scripts/merge_trajectory_runs.py runs/h2-trajectory-repair_b-easy_medium_20260330_142500")
        sys.exit(1)

    new_run = Path(sys.argv[1])
    if not new_run.exists():
        print(f"ERROR: {new_run} does not exist")
        sys.exit(1)

    # Verify merged dir exists (should have been created by the filtering step)
    if not MERGED_DIR.exists():
        print(f"ERROR: {MERGED_DIR} does not exist. Run the filtering step first.")
        sys.exit(1)

    # 1. Check for task_id collisions
    existing_ids = set()
    with open(MERGED_DIR / "attempts.jsonl", encoding="utf-8") as f:
        for line in f:
            existing_ids.add(json.loads(line)["task_id"])

    new_ids = set()
    new_attempts = []
    with open(new_run / "attempts.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            new_ids.add(rec["task_id"])
            new_attempts.append(line)

    overlap = existing_ids & new_ids
    if overlap:
        print(f"ERROR: {len(overlap)} task_id collisions detected!")
        print(f"  Examples: {list(overlap)[:5]}")
        print("  These tasks already exist in the merged folder.")
        sys.exit(1)

    # 2. Append attempts.jsonl
    with open(MERGED_DIR / "attempts.jsonl", "a", encoding="utf-8") as f:
        for line in new_attempts:
            f.write(line)
    print(f"Appended {len(new_attempts)} attempts ({len(new_ids)} tasks)")

    # 3. Copy hidden_states/*.npz
    new_hs = new_run / "hidden_states"
    merged_hs = MERGED_DIR / "hidden_states"
    if new_hs.exists():
        copied = 0
        for npz in sorted(new_hs.glob("*.npz")):
            dest = merged_hs / npz.name
            if dest.exists():
                print(f"  WARNING: {npz.name} already exists in merged, skipping")
                continue
            shutil.copy2(npz, dest)
            copied += 1
        print(f"Copied {copied} hidden_state files")
    else:
        print("WARNING: no hidden_states/ directory in new run")

    # 4. Summary
    total_attempts = sum(1 for _ in open(MERGED_DIR / "attempts.jsonl", encoding="utf-8"))
    total_npz = len(list(merged_hs.glob("*.npz")))
    print(f"\nMerged dataset now has:")
    print(f"  {total_attempts} total attempt records")
    print(f"  {total_npz} hidden_state files (tasks)")


if __name__ == "__main__":
    main()
