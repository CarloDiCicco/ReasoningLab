"""Cross-policy invariant tests.

All three policies share the same public contract (same function signature, same
PolicyResult shape).  These tests verify invariants that must hold for every
policy — catching regressions where one policy violates a shared expectation.

Strategy:
  - Parametrize over all three (policy_fn, policy_module, policy_label) tuples.
  - Use conftest fakes + monkeypatching (same pattern as individual policy tests).
  - Each test checks one invariant that is meaningful to verify across policies.

This file is NOT a substitute for the per-policy test files.  It is a thin
cross-cutting net that catches cases where a policy changes its record format
in a way that breaks the H1 comparison (e.g., wrong task_id, wrong policy label,
non-sequential attempt_idx).
"""

from __future__ import annotations

import pytest

import reasoninglab.policies.baseline  as baseline_module
import reasoninglab.policies.best_of_b as best_of_b_module
import reasoninglab.policies.repair_b  as repair_b_module
from reasoninglab.policies.baseline  import run_baseline
from reasoninglab.policies.best_of_b import run_best_of_b
from reasoninglab.policies.repair_b  import run_repair_b
from reasoninglab.verify.taxonomy import FailureType

from conftest import FakeModel, make_execution_result, make_gen, make_task


# ── Shared parametrize fixture ────────────────────────────────────────────────

# Each tuple: (policy_fn, policy_module, canonical_label_string).
_POLICIES = [
    (run_baseline,  baseline_module,  "baseline"),
    (run_best_of_b, best_of_b_module, "best_of_b"),
    (run_repair_b,  repair_b_module,  "repair_b"),
]


def _patch_all_fail(monkeypatch: pytest.MonkeyPatch, module, budget_B: int):
    """Monkeypatch executor and classifier so every attempt fails.

    Returns a FakeModel pre-loaded with budget_B generations.
    """
    monkeypatch.setattr(
        module, "execute_candidate",
        lambda *_args, **_kwargs: make_execution_result(passed=False, stderr="err"),
    )
    monkeypatch.setattr(
        module, "classify_result",
        lambda _result: FailureType.ASSERTION,
    )
    return FakeModel([make_gen() for _ in range(budget_B)])


def _patch_pass_on_first(monkeypatch: pytest.MonkeyPatch, module):
    """Monkeypatch so the first attempt always passes.

    Returns a FakeModel with one generation.
    """
    monkeypatch.setattr(
        module, "execute_candidate",
        lambda *_args, **_kwargs: make_execution_result(passed=True),
    )
    monkeypatch.setattr(
        module, "classify_result",
        lambda _result: FailureType.PASS,
    )
    return FakeModel([make_gen(text="def solve(): return 1")])


# ── Tests: AttemptRecord field invariants ─────────────────────────────────────

@pytest.mark.parametrize("policy_fn,module,label", _POLICIES)
def test_all_policies_set_task_id_on_every_record(
    monkeypatch: pytest.MonkeyPatch, policy_fn, module, label: str
):
    """Every AttemptRecord produced by every policy must carry the correct task_id."""
    task = make_task()
    model = _patch_all_fail(monkeypatch, module, budget_B=3)
    result = policy_fn(task=task, model=model, budget_B=3)
    assert all(a.task_id == task.task_id for a in result.attempts)


@pytest.mark.parametrize("policy_fn,module,label", _POLICIES)
def test_all_policies_set_policy_field_correctly(
    monkeypatch: pytest.MonkeyPatch, policy_fn, module, label: str
):
    """AttemptRecord.policy must equal the canonical label for that policy."""
    model = _patch_all_fail(monkeypatch, module, budget_B=2)
    result = policy_fn(task=make_task(), model=model, budget_B=2)
    assert all(a.policy == label for a in result.attempts)


@pytest.mark.parametrize("policy_fn,module,label", _POLICIES)
def test_all_policies_attempt_idx_starts_at_zero(
    monkeypatch: pytest.MonkeyPatch, policy_fn, module, label: str
):
    """The first AttemptRecord must always have attempt_idx == 0."""
    model = _patch_all_fail(monkeypatch, module, budget_B=3)
    result = policy_fn(task=make_task(), model=model, budget_B=3)
    assert result.attempts[0].attempt_idx == 0


@pytest.mark.parametrize("policy_fn,module,label", _POLICIES)
def test_all_policies_attempt_idx_is_sequential(
    monkeypatch: pytest.MonkeyPatch, policy_fn, module, label: str
):
    """attempt_idx values must be consecutive integers starting from 0."""
    budget_B = 3
    model = _patch_all_fail(monkeypatch, module, budget_B=budget_B)

    # Baseline uses only 1 call regardless of budget_B; the others use all 3.
    if policy_fn is run_baseline:
        result = policy_fn(task=make_task(), model=model, budget_B=budget_B)
        expected = [0]
    else:
        result = policy_fn(task=make_task(), model=model, budget_B=budget_B)
        expected = list(range(budget_B))

    assert [a.attempt_idx for a in result.attempts] == expected


# ── Tests: FailureType / pass invariants ─────────────────────────────────────

@pytest.mark.parametrize("policy_fn,module,label", _POLICIES)
def test_all_policies_pass_type_on_passing_attempt(
    monkeypatch: pytest.MonkeyPatch, policy_fn, module, label: str
):
    """When an attempt passes, its failure_type must be FailureType.PASS."""
    model = _patch_pass_on_first(monkeypatch, module)
    result = policy_fn(task=make_task(), model=model, budget_B=5)
    # At least one attempt must have passed.
    passing = [a for a in result.attempts if a.passed]
    assert passing, "expected at least one passing attempt"
    assert all(a.failure_type == FailureType.PASS for a in passing)


# ── Tests: selected_candidate invariants ─────────────────────────────────────

@pytest.mark.parametrize("policy_fn,module,label", _POLICIES)
def test_all_policies_selected_candidate_set_on_pass(
    monkeypatch: pytest.MonkeyPatch, policy_fn, module, label: str
):
    """selected_candidate must not be None when at least one attempt passed."""
    model = _patch_pass_on_first(monkeypatch, module)
    result = policy_fn(task=make_task(), model=model, budget_B=5)
    assert result.selected_candidate is not None


@pytest.mark.parametrize("policy_fn,module,label", _POLICIES)
def test_all_policies_selected_candidate_none_on_all_fail(
    monkeypatch: pytest.MonkeyPatch, policy_fn, module, label: str
):
    """selected_candidate must be None when every attempt fails."""
    budget_B = 3
    model = _patch_all_fail(monkeypatch, module, budget_B=budget_B)
    result = policy_fn(task=make_task(), model=model, budget_B=budget_B)
    assert result.selected_candidate is None


# ── Tests: attempts is always a tuple ────────────────────────────────────────

@pytest.mark.parametrize("policy_fn,module,label", _POLICIES)
def test_all_policies_return_attempts_as_tuple(
    monkeypatch: pytest.MonkeyPatch, policy_fn, module, label: str
):
    """PolicyResult.attempts must be a tuple (immutable, hashable) for all policies."""
    model = _patch_all_fail(monkeypatch, module, budget_B=2)
    result = policy_fn(task=make_task(), model=model, budget_B=2)
    assert isinstance(result.attempts, tuple)
