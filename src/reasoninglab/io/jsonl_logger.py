"""
io/jsonl_logger.py
──────────────────
Writes structured experiment records to disk in JSONL format
(one JSON object per line, newline-delimited).

JSONL is the standard format for ML experiment logging because:
  - Append-only: each attempt is written immediately, so a crash mid-run
    does not lose previously completed attempts.
  - Streamable: results can be read line-by-line without loading the full
    file into memory.
  - Human-readable: plain text, easy to inspect with any text editor or
    grep/jq from the command line.

Two public functions:
  - log_attempt(path, record)  : appends one AttemptRecord line
  - log_summary(path, metrics) : appends one RunMetrics line

File layout produced by a full run:
  results/
    <run_id>/
      attempts.jsonl   ← one line per attempt, written during the run
      summary.jsonl    ← one line per policy arm, written after the run

Each line in attempts.jsonl looks like:
  {"task_id": "HumanEval/0", "policy": "repair_b", "attempt_idx": 0,
   "passed": false, "failure_type": "assertion", "elapsed_s": 2.41,
   "tokens": 312}

Each line in summary.jsonl looks like:
  {"policy": "repair_b", "pass_rate": 0.72, "mean_elapsed_s": 3.1,
   "mean_tokens": 287.4, "total_attempts": 150, "total_tasks": 50,
   "failure_dist": {"assertion": 0.61, "syntax": 0.21, "runtime": 0.18}}

Consumed by:
  - eval/runner.py    : calls log_attempt() inside the policy loop,
                        calls log_summary() once per arm after all tasks
  - scripts/plot_h1.py: reads the JSONL files to produce plots and tables
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from reasoninglab.eval.metrics import AttemptRecord, RunMetrics


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _append_line(path: Path, obj: dict) -> None:
    """Serialize obj to a single JSON line and append it to path.

    Creates the file (and any missing parent directories) if they do not
    exist yet.  Uses append mode ("a") so existing lines are never touched —
    a crash mid-run leaves all previously written lines intact.

    Args:
        path: Destination .jsonl file path.
        obj:  A plain dict that is fully JSON-serializable.
    """
    # Create parent directories (e.g. results/<run_id>/) on first write.
    path.parent.mkdir(parents=True, exist_ok=True)

    # json.dumps produces a single-line string (no newline inside).
    # We add "\n" explicitly so each call appends exactly one complete line.
    # ensure_ascii=False preserves any Unicode in task prompts or model output.
    line = json.dumps(obj, ensure_ascii=False) + "\n"

    with path.open("a", encoding="utf-8") as f:
        f.write(line)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def log_attempt(path: Path, record: AttemptRecord) -> None:
    """Append one attempt record as a JSON line to the given file.

    Called once per attempt inside the policy run loop, immediately after
    execute_candidate() and classify_result() return.  Writing immediately
    (rather than buffering all records) ensures data is not lost if the
    runner crashes or is interrupted mid-run.

    The AttemptRecord dataclass is converted to a plain dict via
    dataclasses.asdict().  Because FailureType inherits from str, its value
    serializes as a plain string (e.g. "assertion") with no custom encoder.

    Args:
        path:   Path to the attempts JSONL file (e.g. results/<run>/attempts.jsonl).
        record: The AttemptRecord produced by the policy for this attempt.
    """
    _append_line(path, dataclasses.asdict(record))


def log_summary(path: Path, policy: str, metrics: RunMetrics) -> None:
    """Append one run summary as a JSON line to the given file.

    Called once per policy arm after all tasks have been processed.
    The policy name is included as a field so multiple arms can be written
    to the same summary.jsonl file and distinguished when reading.

    RunMetrics is not a simple flat dataclass (failure_dist is a nested dict),
    so we build the output dict manually rather than using dataclasses.asdict(),
    which would recursively convert nested objects in ways we don't need here.

    Args:
        path:    Path to the summary JSONL file (e.g. results/<run>/summary.jsonl).
        policy:  Policy name string ("baseline" | "best_of_b" | "repair_b").
        metrics: The RunMetrics produced by compute_metrics() for this arm.
    """
    obj = {
        "policy":        policy,
        "pass_rate":     metrics.pass_rate,
        "mean_elapsed_s": metrics.mean_elapsed_s,
        "mean_tokens":   metrics.mean_tokens,
        "total_attempts": metrics.total_attempts,
        "total_tasks":   metrics.total_tasks,
        "failure_dist":  metrics.failure_dist,   # already dict[str, float]
    }
    _append_line(path, obj)
