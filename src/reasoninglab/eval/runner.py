"""
eval/runner.py
──────────────
Orchestrates one policy arm across a full task list: calls the policy for each
task, logs every attempt immediately, aggregates metrics, and writes the summary.

This module is the glue between the pieces built in the previous steps:

    run_experiment()
      ├─ for each task → policy_fn(task, model, budget_B, ...) → PolicyResult
      │     └─ log_attempt() per record (append-only, crash-safe)
      └─ compute_metrics(all_records) → RunMetrics
            └─ log_summary() once

Design choices:
  - policy_fn is a Callable, not a string — the caller (cli.py) resolves the
    string to a function.  This keeps runner.py free of policy dispatch logic
    and makes it trivially testable with any stub.
  - policy_name is a separate str used only for the JSONL summary line.
  - attempts_path / summary_path are full Paths, not a run_id; naming
    conventions live in cli.py.
  - verifier_timeout_s=None is forwarded verbatim; the policy resolves the
    fallback to task.timeout_s.
  - Exceptions from the policy propagate unchanged — fail fast in a research
    harness rather than silently swallowing errors.

Consumed by:
  - cli.py : calls run_experiment() once per policy arm
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from reasoninglab.eval.metrics import AttemptRecord, RunMetrics, compute_metrics
from reasoninglab.io.jsonl_logger import log_attempt, log_summary
from reasoninglab.policies.contracts import PolicyResult, SupportsGenerate
from reasoninglab.tasks.schema import TaskRecord


def run_experiment(
    tasks: list[TaskRecord],
    model: SupportsGenerate,
    policy_fn: Callable[..., PolicyResult], #policy_fn is a variable expected to hold a function, the function can take any argument (...), that function must return a PolicyResult
    policy_name: str,
    budget_B: int,
    attempts_path: Path,
    summary_path: Path,
    *,
    verifier_timeout_s: float | None = None,
    return_hidden_states: bool = False,
    extra_summary_fields: dict | None = None,
) -> RunMetrics:
    """Run one policy arm over all tasks, logging attempts and returning metrics.

    For each task, calls policy_fn once with the task, model, and budget
    parameters.  Each AttemptRecord from the result is written to attempts_path
    immediately (append mode) so data is not lost if the runner crashes
    mid-run.  After all tasks, compute_metrics() aggregates the records and
    log_summary() appends one summary line to summary_path.

    Args:
        tasks:                Ordered list of tasks to evaluate.  Must be non-empty.
        model:                Model adapter implementing SupportsGenerate.
        policy_fn:            One of run_baseline, run_best_of_b, run_repair_b —
                              or any callable with the same keyword-argument signature.
        policy_name:          Label written to the summary JSONL line
                              ("baseline" | "best_of_b" | "repair_b").
        budget_B:             Maximum model calls per task, forwarded to policy_fn.
        attempts_path:        Path to append one JSON line per attempt.
                              Created (with parent directories) if absent.
        summary_path:         Path to append the aggregate summary line.
                              Created (with parent directories) if absent.
        verifier_timeout_s:   Optional per-attempt timeout override forwarded to
                              policy_fn.  None means each policy uses task.timeout_s.
        return_hidden_states: Whether to request hidden states from the model,
                              forwarded to policy_fn unchanged.
        extra_summary_fields: Optional dict of additional fields to include in
                              the summary JSONL line (e.g. model_id, tasks_file).

    Returns:
        RunMetrics aggregated across all tasks and all their attempts.

    Raises:
        ValueError: if tasks is empty (no data to evaluate or aggregate).
    """
    if not tasks:
        raise ValueError(
            "run_experiment received an empty tasks list — nothing to evaluate."
        )

    all_attempts: list[AttemptRecord] = []

    for task in tasks:
        # Call the policy for this task.  The policy handles the generate→execute→
        # classify loop internally; we only see the finished PolicyResult.
        result: PolicyResult = policy_fn(
            task=task,
            model=model,
            budget_B=budget_B,
            verifier_timeout_s=verifier_timeout_s,
            return_hidden_states=return_hidden_states,
        )

        # Write each attempt immediately so a crash mid-run preserves prior data.
        for record in result.attempts:
            all_attempts.append(record)
            log_attempt(attempts_path, record)

    # Aggregate after all tasks so metrics are consistent across the whole run.
    metrics = compute_metrics(all_attempts)
    log_summary(summary_path, policy_name, metrics, extra=extra_summary_fields)
    return metrics
