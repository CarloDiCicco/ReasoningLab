"""
tests/test_metrics.py
─────────────────────
Unit tests for eval/metrics.py.

Tests cover:
  - AttemptRecord construction and immutability
  - compute_metrics() pass_rate is task-level (not attempt-level)
  - compute_metrics() failure_dist includes only failed attempts, sums to 1.0
  - compute_metrics() mean_elapsed_s and mean_tokens are correct
  - compute_metrics() total_attempts and total_tasks counts
  - Edge cases: all pass, all fail, single record, empty raises ValueError
"""

from __future__ import annotations

import dataclasses

import pytest

from reasoninglab.eval.metrics import AttemptRecord, RunMetrics, compute_metrics
from reasoninglab.verify.taxonomy import FailureType


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _attempt(
    task_id: str = "t1",
    policy: str = "baseline",
    attempt_idx: int = 0,
    passed: bool = False,
    failure_type: FailureType = FailureType.ASSERTION,
    elapsed_s: float = 1.0,
    tokens: int = 100,
) -> AttemptRecord:
    """Construct an AttemptRecord with sensible defaults for test brevity."""
    return AttemptRecord(
        task_id=task_id,
        policy=policy,
        attempt_idx=attempt_idx,
        passed=passed,
        failure_type=failure_type,
        elapsed_s=elapsed_s,
        tokens=tokens,
    )


# ─────────────────────────────────────────────────────────────────────────────
# AttemptRecord
# ─────────────────────────────────────────────────────────────────────────────

class TestAttemptRecord:

    def test_construction(self):
        r = _attempt(task_id="abc", passed=True, tokens=200)
        assert r.task_id == "abc"
        assert r.passed is True
        assert r.tokens == 200

    def test_frozen(self):
        """AttemptRecord must be immutable after construction."""
        r = _attempt()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            r.passed = True  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# compute_metrics — pass_rate
# ─────────────────────────────────────────────────────────────────────────────

class TestPassRate:

    def test_all_pass(self):
        records = [
            _attempt(task_id="t1", passed=True, failure_type=FailureType.PASS),
            _attempt(task_id="t2", passed=True, failure_type=FailureType.PASS),
        ]
        m = compute_metrics(records)
        assert m.pass_rate == 1.0

    def test_all_fail(self):
        records = [
            _attempt(task_id="t1", passed=False),
            _attempt(task_id="t2", passed=False),
        ]
        m = compute_metrics(records)
        assert m.pass_rate == 0.0

    def test_half_pass(self):
        records = [
            _attempt(task_id="t1", passed=True, failure_type=FailureType.PASS),
            _attempt(task_id="t2", passed=False),
        ]
        m = compute_metrics(records)
        assert m.pass_rate == pytest.approx(0.5)

    def test_pass_rate_is_task_level_not_attempt_level(self):
        """Critical: a task with 2 failing attempts + 1 passing attempt counts
        as 1 passed task, not as 0.33 pass rate.  This ensures the metric is
        not inflated for multi-attempt policies vs baseline."""
        records = [
            # Task t1: 2 fails then 1 pass → task PASSES
            _attempt(task_id="t1", attempt_idx=0, passed=False),
            _attempt(task_id="t1", attempt_idx=1, passed=False),
            _attempt(task_id="t1", attempt_idx=2, passed=True, failure_type=FailureType.PASS),
            # Task t2: 2 fails → task FAILS
            _attempt(task_id="t2", attempt_idx=0, passed=False),
            _attempt(task_id="t2", attempt_idx=1, passed=False),
        ]
        m = compute_metrics(records)
        # 1 out of 2 tasks passed
        assert m.pass_rate == pytest.approx(0.5)
        assert m.total_tasks == 2
        assert m.total_attempts == 5

    def test_single_passing_record(self):
        records = [_attempt(passed=True, failure_type=FailureType.PASS)]
        m = compute_metrics(records)
        assert m.pass_rate == 1.0
        assert m.total_tasks == 1
        assert m.total_attempts == 1

    def test_single_failing_record(self):
        records = [_attempt(passed=False)]
        m = compute_metrics(records)
        assert m.pass_rate == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_metrics — failure_dist
# ─────────────────────────────────────────────────────────────────────────────

class TestFailureDist:

    def test_empty_when_all_pass(self):
        """No failed attempts → failure_dist must be empty, not a dict of zeros."""
        records = [
            _attempt(task_id="t1", passed=True, failure_type=FailureType.PASS),
            _attempt(task_id="t2", passed=True, failure_type=FailureType.PASS),
        ]
        m = compute_metrics(records)
        assert m.failure_dist == {}

    def test_single_failure_type(self):
        records = [
            _attempt(task_id="t1", passed=False, failure_type=FailureType.SYNTAX),
            _attempt(task_id="t2", passed=False, failure_type=FailureType.SYNTAX),
        ]
        m = compute_metrics(records)
        assert m.failure_dist == {"syntax": pytest.approx(1.0)}

    def test_two_failure_types_equal_split(self):
        records = [
            _attempt(task_id="t1", passed=False, failure_type=FailureType.SYNTAX),
            _attempt(task_id="t2", passed=False, failure_type=FailureType.ASSERTION),
        ]
        m = compute_metrics(records)
        assert m.failure_dist["syntax"]    == pytest.approx(0.5)
        assert m.failure_dist["assertion"] == pytest.approx(0.5)

    def test_failure_dist_sums_to_one(self):
        """The failure_dist fractions must sum to 1.0 for any mix of failure types."""
        records = [
            _attempt(task_id="t1", passed=False, failure_type=FailureType.SYNTAX),
            _attempt(task_id="t2", passed=False, failure_type=FailureType.ASSERTION),
            _attempt(task_id="t3", passed=False, failure_type=FailureType.RUNTIME),
            _attempt(task_id="t4", passed=False, failure_type=FailureType.TIMEOUT),
            _attempt(task_id="t5", passed=False, failure_type=FailureType.UNKNOWN),
        ]
        m = compute_metrics(records)
        assert sum(m.failure_dist.values()) == pytest.approx(1.0)

    def test_passing_attempts_excluded_from_failure_dist(self):
        """Passing attempts must NOT appear in failure_dist, even though their
        failure_type field is FailureType.PASS."""
        records = [
            _attempt(task_id="t1", passed=True,  failure_type=FailureType.PASS),
            _attempt(task_id="t2", passed=False, failure_type=FailureType.SYNTAX),
            _attempt(task_id="t3", passed=False, failure_type=FailureType.SYNTAX),
        ]
        m = compute_metrics(records)
        # Only 2 failed attempts, both SYNTAX → 100% syntax
        assert m.failure_dist == {"syntax": pytest.approx(1.0)}
        assert "pass" not in m.failure_dist

    def test_failure_dist_keys_are_strings(self):
        """Keys must be plain strings (FailureType.value), not FailureType objects,
        for clean JSON serialization in the logger."""
        records = [_attempt(passed=False, failure_type=FailureType.RUNTIME)]
        m = compute_metrics(records)
        for key in m.failure_dist:
            assert isinstance(key, str)
            assert not isinstance(key, type)  # not a class


# ─────────────────────────────────────────────────────────────────────────────
# compute_metrics — cost averages
# ─────────────────────────────────────────────────────────────────────────────

class TestCostAverages:

    def test_mean_elapsed_s(self):
        records = [
            _attempt(task_id="t1", elapsed_s=2.0),
            _attempt(task_id="t2", elapsed_s=4.0),
        ]
        m = compute_metrics(records)
        assert m.mean_elapsed_s == pytest.approx(3.0)

    def test_mean_tokens(self):
        records = [
            _attempt(task_id="t1", tokens=100),
            _attempt(task_id="t2", tokens=300),
        ]
        m = compute_metrics(records)
        assert m.mean_tokens == pytest.approx(200.0)

    def test_averages_include_all_attempts(self):
        """Averages are over ALL attempts, not per-task.  A task with 3 attempts
        contributes 3 values to the mean, not 1."""
        records = [
            _attempt(task_id="t1", attempt_idx=0, elapsed_s=1.0, tokens=100),
            _attempt(task_id="t1", attempt_idx=1, elapsed_s=3.0, tokens=300),
        ]
        m = compute_metrics(records)
        assert m.mean_elapsed_s == pytest.approx(2.0)   # (1+3)/2
        assert m.mean_tokens    == pytest.approx(200.0)  # (100+300)/2
        assert m.total_attempts == 2
        assert m.total_tasks    == 1


# ─────────────────────────────────────────────────────────────────────────────
# compute_metrics — edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_metrics([])

    def test_single_record_all_fields_correct(self):
        r = _attempt(
            task_id="solo", passed=False,
            failure_type=FailureType.TIMEOUT,
            elapsed_s=8.1, tokens=512,
        )
        m = compute_metrics([r])
        assert m.pass_rate      == 0.0
        assert m.mean_elapsed_s == pytest.approx(8.1)
        assert m.mean_tokens    == pytest.approx(512.0)
        assert m.total_attempts == 1
        assert m.total_tasks    == 1
        assert m.failure_dist   == {"timeout": pytest.approx(1.0)}
