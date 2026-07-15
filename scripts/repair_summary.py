#!/usr/bin/env python3
"""Repair-trajectory summary: how often does a failed first attempt ever recover?

Question this answers: within the repair budget, how many tasks whose first
attempt fails are eventually solved, at which repair step, and how does the
pass rate among still-unsolved tasks evolve across attempts? These counts are
what the paper reports as its null result -- successful repairs are too rare to
estimate a contrastive direction in hidden-state space.

Reads attempts.jsonl only (no hidden states, no model, no sklearn).

Usage:
    python scripts/repair_summary.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_outcomes(run_dir: Path) -> dict[str, dict[int, bool]]:
    """Map task_id -> {attempt_idx: passed} from the run's attempts.jsonl."""
    outcomes: dict[str, dict[int, bool]] = defaultdict(dict)
    with open(run_dir / "attempts.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            outcomes[rec["task_id"]][rec["attempt_idx"]] = bool(rec["passed"])
    return outcomes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs/h2-trajectory")
    p.add_argument("--output-dir", default="results/h2/repair_summary")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[repair-summary] Loading attempts from {args.run_dir} ...")
    outcomes = load_outcomes(Path(args.run_dir))
    tasks = list(outcomes)
    # Budget comes from the data, not the config, so this stays correct if the
    # run is regenerated with a different budget_B.
    budget = max(len(v) for v in outcomes.values())

    passed_0 = [t for t in tasks if outcomes[t].get(0) is True]
    failed_0 = [t for t in tasks if outcomes[t].get(0) is False]
    recovered = [t for t in failed_0
                 if any(outcomes[t].get(i) for i in range(1, budget))]

    # Successful repair transitions: a failed attempt followed by a passing one.
    transitions: Counter[str] = Counter()
    for t in failed_0:
        for i in range(1, budget):
            if outcomes[t].get(i - 1) is False and outcomes[t].get(i) is True:
                transitions[f"{i-1}->{i}"] += 1

    # Conditional pass rate: among tasks still unsolved entering each attempt,
    # what fraction does that attempt solve? Attempt 0's denominator is all tasks.
    conditional: dict[str, dict] = {
        "0": {"passed": len(passed_0), "eligible": len(tasks),
              "rate": len(passed_0) / len(tasks)}
    }
    unsolved = set(failed_0)
    for i in range(1, budget):
        solved_now = {t for t in unsolved if outcomes[t].get(i)}
        conditional[str(i)] = {
            "passed": len(solved_now),
            "eligible": len(unsolved),
            "rate": (len(solved_now) / len(unsolved)) if unsolved else 0.0,
        }
        unsolved -= solved_now

    print()
    print("=" * 60)
    print(f"REPAIR SUMMARY  budget={budget}")
    print("=" * 60)
    print(f"  Tasks              : {len(tasks)}")
    print(f"  Attempt-0 pass     : {len(passed_0)} ({len(passed_0)/len(tasks):.1%})")
    print(f"  Attempt-0 fail     : {len(failed_0)}")
    print(f"  Ever recovered     : {len(recovered)} of {len(failed_0)}")
    print(f"  Repair transitions : {dict(sorted(transitions.items()))} "
          f"(total {sum(transitions.values())})")
    print()
    print("  Conditional pass rate among still-unsolved tasks:")
    for i in range(budget):
        c = conditional[str(i)]
        print(f"    attempt {i}: {c['passed']}/{c['eligible']} = {c['rate']:.1%}")
    print("=" * 60)

    payload = {
        "run_dir": str(args.run_dir),
        "budget": budget,
        "n_tasks": len(tasks),
        "attempt0_pass": len(passed_0),
        "attempt0_pass_rate": len(passed_0) / len(tasks),
        "attempt0_fail": len(failed_0),
        "ever_recovered": len(recovered),
        "repair_transitions": dict(sorted(transitions.items())),
        "repair_transitions_total": sum(transitions.values()),
        "conditional_pass_rate": conditional,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[repair-summary] Saved {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
