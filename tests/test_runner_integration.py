"""Integration tests for eval/runner.py.

run_experiment() is a thin orchestration layer — its interesting behaviour is
entirely in the wiring: does it call the policy for every task, forward the
right arguments, log every record immediately, and hand back correct metrics?

Strategy:
  - Fake policy_fn returns pre-set PolicyResults in task order.
  - log_attempt / log_summary are monkeypatched on the runner module so no
    filesystem I/O happens during tests (same pattern as policy tests).
  - Fake paths (Path("attempts.jsonl") / Path("summary.jsonl")) are passed as
    arguments but never actually touched because the loggers are stubbed.
  - FakeModel is passed through to policy_fn but the fake policy_fn ignores it;
    the model is present only to satisfy the type signature.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import reasoninglab.eval.runner as runner_module
from reasoninglab.eval.runner import run_experiment
from reasoninglab.eval.metrics import RunMetrics
from reasoninglab.policies.contracts import PolicyResult
from reasoninglab.verify.taxonomy import FailureType

from conftest import FakeModel, make_execution_result, make_gen, make_task
from reasoninglab.eval.metrics import AttemptRecord


# ── Helpers ───────────────────────────────────────────────────────────────────

# Placeholder paths — never hit disk because log_attempt / log_summary are stubbed.
_ATTEMPTS_PATH = Path("attempts.jsonl")
_SUMMARY_PATH  = Path("summary.jsonl")


def _make_attempt(task_id: str, passed: bool, attempt_idx: int = 0) -> AttemptRecord:
    """Create a minimal AttemptRecord for runner tests."""
    return AttemptRecord(
        task_id=task_id,
        policy="fake_policy",
        attempt_idx=attempt_idx,
        passed=passed,
        failure_type=FailureType.PASS if passed else FailureType.ASSERTION,
        elapsed_s=0.5,
        tokens=10,
    )


def _run_with_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    # Each element is the PolicyResult the fake policy should return for task[i].
    per_task_results: list[PolicyResult],
    policy_name: str = "fake_policy",
    budget_B: int = 3,
    verifier_timeout_s: float | None = None,
    return_hidden_states: bool = False,
):
    """Run run_experiment with monkeypatched loggers and a fake policy_fn.

    Returns (metrics, logged_attempts, logged_summary_calls) where:
      - logged_attempts   : list of AttemptRecord objects passed to log_attempt
      - logged_summary_calls: list of (policy_name, RunMetrics) tuples
    """
    tasks = [make_task(prompt=f"Task {i}") for i in range(len(per_task_results))]

    # Fake policy: pops results in order, records what arguments it received.
    call_log: list[dict] = []
    result_iter = iter(per_task_results)

    def _fake_policy_fn(**kwargs) -> PolicyResult:
        call_log.append(kwargs)
        return next(result_iter)

    # Stub loggers to capture calls without touching the filesystem.
    logged_attempts: list[AttemptRecord] = []
    logged_summary_calls: list[tuple[str, RunMetrics]] = []

    monkeypatch.setattr(runner_module, "log_attempt", lambda _path, record: logged_attempts.append(record))
    monkeypatch.setattr(runner_module, "log_summary", lambda _path, name, m: logged_summary_calls.append((name, m)))

    model = FakeModel([])  # not called — policy_fn is fully stubbed

    metrics = run_experiment(
        tasks=tasks,
        model=model,
        policy_fn=_fake_policy_fn,
        policy_name=policy_name,
        budget_B=budget_B,
        attempts_path=_ATTEMPTS_PATH,
        summary_path=_SUMMARY_PATH,
        verifier_timeout_s=verifier_timeout_s,
        return_hidden_states=return_hidden_states,
    )
    return metrics, logged_attempts, logged_summary_calls, call_log


# ── Tests: task processing ────────────────────────────────────────────────────

def test_run_experiment_processes_all_tasks(monkeypatch: pytest.MonkeyPatch):
    # policy_fn must be called exactly once per task, in order.
    tasks_n = 4
    per_task_results = [
        PolicyResult(attempts=(_make_attempt(f"task-{i}", passed=False),), selected_candidate=None)
        for i in range(tasks_n)
    ]
    _, _, _, call_log = _run_with_stubs(monkeypatch, per_task_results=per_task_results)
    assert len(call_log) == tasks_n


def test_run_experiment_empty_tasks_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(runner_module, "log_attempt", lambda *_: None)
    monkeypatch.setattr(runner_module, "log_summary", lambda *_: None)
    with pytest.raises(ValueError):
        run_experiment(
            tasks=[],
            model=FakeModel([]),
            policy_fn=lambda **_: None,
            policy_name="x",
            budget_B=1,
            attempts_path=_ATTEMPTS_PATH,
            summary_path=_SUMMARY_PATH,
        )


# ── Tests: logging ────────────────────────────────────────────────────────────

def test_run_experiment_logs_every_attempt(monkeypatch: pytest.MonkeyPatch):
    # With 3 tasks × 2 attempts each, log_attempt must be called 6 times total.
    per_task_results = [
        PolicyResult(
            attempts=(
                _make_attempt(f"task-{i}", passed=False, attempt_idx=0),
                _make_attempt(f"task-{i}", passed=True,  attempt_idx=1),
            ),
            selected_candidate="def solve(): pass",
        )
        for i in range(3)
    ]
    _, logged_attempts, _, _ = _run_with_stubs(monkeypatch, per_task_results=per_task_results)
    assert len(logged_attempts) == 6  # 3 tasks × 2 attempts


def test_run_experiment_logs_summary_once(monkeypatch: pytest.MonkeyPatch):
    # log_summary is called exactly once, after all tasks.
    per_task_results = [
        PolicyResult(attempts=(_make_attempt("t", passed=True),), selected_candidate="x")
    ]
    _, _, summary_calls, _ = _run_with_stubs(monkeypatch, per_task_results=per_task_results)
    assert len(summary_calls) == 1


def test_run_experiment_logs_summary_with_correct_policy_name(monkeypatch: pytest.MonkeyPatch):
    per_task_results = [
        PolicyResult(attempts=(_make_attempt("t", passed=True),), selected_candidate="x")
    ]
    _, _, summary_calls, _ = _run_with_stubs(
        monkeypatch,
        per_task_results=per_task_results,
        policy_name="best_of_b",
    )
    assert summary_calls[0][0] == "best_of_b"


def test_run_experiment_attempt_records_written_in_order(monkeypatch: pytest.MonkeyPatch):
    # Records from task 0 must appear before records from task 1 in the log.
    per_task_results = [
        PolicyResult(attempts=(_make_attempt("task-0", passed=False),), selected_candidate=None),
        PolicyResult(attempts=(_make_attempt("task-1", passed=True),),  selected_candidate="x"),
    ]
    _, logged_attempts, _, _ = _run_with_stubs(monkeypatch, per_task_results=per_task_results)
    assert logged_attempts[0].task_id == "task-0"
    assert logged_attempts[1].task_id == "task-1"


# ── Tests: metrics ────────────────────────────────────────────────────────────

def test_run_experiment_returns_correct_pass_rate(monkeypatch: pytest.MonkeyPatch):
    # 2 out of 4 tasks pass → pass_rate = 0.5.
    per_task_results = [
        PolicyResult(attempts=(_make_attempt("t0", passed=True),),  selected_candidate="x"),
        PolicyResult(attempts=(_make_attempt("t1", passed=False),), selected_candidate=None),
        PolicyResult(attempts=(_make_attempt("t2", passed=True),),  selected_candidate="x"),
        PolicyResult(attempts=(_make_attempt("t3", passed=False),), selected_candidate=None),
    ]
    metrics, _, _, _ = _run_with_stubs(monkeypatch, per_task_results=per_task_results)
    assert metrics.pass_rate == pytest.approx(0.5)


# ── Tests: argument forwarding ────────────────────────────────────────────────

def test_run_experiment_passes_budget_B_to_policy(monkeypatch: pytest.MonkeyPatch):
    per_task_results = [
        PolicyResult(attempts=(_make_attempt("t", passed=True),), selected_candidate="x")
    ]
    _, _, _, call_log = _run_with_stubs(monkeypatch, per_task_results=per_task_results, budget_B=7)
    assert call_log[0]["budget_B"] == 7


def test_run_experiment_passes_verifier_timeout_to_policy(monkeypatch: pytest.MonkeyPatch):
    per_task_results = [
        PolicyResult(attempts=(_make_attempt("t", passed=True),), selected_candidate="x")
    ]
    _, _, _, call_log = _run_with_stubs(
        monkeypatch,
        per_task_results=per_task_results,
        verifier_timeout_s=2.5,
    )
    assert call_log[0]["verifier_timeout_s"] == 2.5


@pytest.mark.parametrize("return_hidden_states", [False, True])
def test_run_experiment_passes_return_hidden_states_to_policy(
    monkeypatch: pytest.MonkeyPatch, return_hidden_states: bool
):
    per_task_results = [
        PolicyResult(attempts=(_make_attempt("t", passed=True),), selected_candidate="x")
    ]
    _, _, _, call_log = _run_with_stubs(
        monkeypatch,
        per_task_results=per_task_results,
        return_hidden_states=return_hidden_states,
    )
    assert call_log[0]["return_hidden_states"] is return_hidden_states
