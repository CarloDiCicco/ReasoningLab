"""Tests for the probe_guided policy.

The probe is stubbed with a FakeProbe that returns a fixed P(pass) score.
The model and verifier are stubbed via monkeypatching, same pattern as the
best_of_b and repair_b test suites.

Key invariants tested:
  - Probe is called exactly once per task (hidden states from attempt 0).
  - P(pass) >= threshold → repair strategy (repair prompts for attempts 1+).
  - P(pass) <  threshold → retry strategy (original prompt for attempts 1+).
  - Early exit on first passing attempt (no unnecessary model calls).
  - Budget exhaustion returns None as selected_candidate.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FakeGeneration, FakeModel, make_execution_result, make_task
from reasoninglab.policies.probe_guided import run_probe_guided
from reasoninglab.verify.executor import execute_candidate


# ── Fake probe ────────────────────────────────────────────────────────────────


class FakeProbe:
    """Sklearn Pipeline stub that returns a fixed P(pass) probability.

    predict_proba is called with a (1, hidden_dim) array and returns a
    (1, 2) array matching sklearn's convention: [[P(fail), P(pass)]].
    """

    def __init__(self, p_pass: float) -> None:
        self.p_pass = p_pass
        self.call_count = 0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.call_count += 1
        # Column 0 = P(fail), column 1 = P(pass).
        return np.array([[1 - self.p_pass, self.p_pass]])


# ── Helpers ───────────────────────────────────────────────────────────────────

_PROBE_LAYER = 35
_HIDDEN_DIM = 16
_FAKE_HIDDEN_STATES = {_PROBE_LAYER: np.zeros(_HIDDEN_DIM, dtype=np.float32)}


def _make_gen_with_hs(text: str = "def f(): pass") -> FakeGeneration:
    """Generation that carries the fake hidden states needed by the probe."""
    return FakeGeneration(
        text=text,
        prompt_tokens=4,
        completion_tokens=3,
        elapsed_s=0.2,
        hidden_states=_FAKE_HIDDEN_STATES,
    )


def _make_gen_no_hs(text: str = "def f(): pass") -> FakeGeneration:
    """Generation without hidden states (for attempts 1+ when not requested)."""
    return FakeGeneration(
        text=text,
        prompt_tokens=4,
        completion_tokens=3,
        elapsed_s=0.2,
        hidden_states=None,
    )


def _run(
    monkeypatch,
    generations: list[FakeGeneration],
    executions: list,
    p_pass: float,
    budget_B: int = 3,
    threshold: float = 0.5,
):
    """Helper: run probe_guided with stubbed model, verifier, and probe."""
    model = FakeModel(generations)
    probe = FakeProbe(p_pass)
    exec_iter = iter(executions)

    monkeypatch.setattr(
        "reasoninglab.policies.probe_guided.execute_candidate",
        lambda **_kw: next(exec_iter),
    )

    result = run_probe_guided(
        task=make_task(),
        model=model,
        budget_B=budget_B,
        verifier_timeout_s=1.0,
        return_hidden_states=False,
        probe=probe,
        probe_layer=_PROBE_LAYER,
        threshold=threshold,
    )
    return result, model, probe


# ── Tests: early exit ─────────────────────────────────────────────────────────


def test_attempt_0_passes_no_further_calls(monkeypatch):
    """If attempt 0 succeeds, the policy returns immediately."""
    result, model, probe = _run(
        monkeypatch,
        generations=[_make_gen_with_hs()],
        executions=[make_execution_result(passed=True)],
        p_pass=0.8,
        budget_B=5,
    )
    assert result.selected_candidate is not None
    assert len(result.attempts) == 1
    # Model was called exactly once (attempt 0 only).
    assert len(model.calls) == 1


# ── Tests: probe called exactly once ─────────────────────────────────────────


def test_probe_called_exactly_once_regardless_of_budget(monkeypatch):
    """The probe scores the prompt once; subsequent attempts don't re-probe."""
    # All attempts fail → uses full budget.
    n_attempts = 4
    result, model, probe = _run(
        monkeypatch,
        generations=[_make_gen_with_hs()] + [_make_gen_no_hs()] * (n_attempts - 1),
        executions=[make_execution_result(passed=False)] * n_attempts,
        p_pass=0.8,
        budget_B=n_attempts,
    )
    # Probe called once regardless of how many model calls were made.
    assert probe.call_count == 1
    assert len(result.attempts) == n_attempts


# ── Tests: routing by probe score ─────────────────────────────────────────────


def test_high_p_pass_uses_repair_prompts(monkeypatch):
    """P(pass)=0.8 >= threshold=0.5 → repair strategy: prompts 1+ include feedback."""
    result, model, probe = _run(
        monkeypatch,
        generations=[_make_gen_with_hs(), _make_gen_no_hs(), _make_gen_no_hs()],
        executions=[
            make_execution_result(passed=False, stderr="AssertionError"),
            make_execution_result(passed=False, stderr="AssertionError"),
            make_execution_result(passed=True),
        ],
        p_pass=0.8,
        budget_B=3,
        threshold=0.5,
    )
    # Attempt 0 used the original prompt; attempts 1 and 2 used repair prompts.
    # Repair prompts contain the original prompt text AND error feedback.
    original_prompt = make_task().prompt
    assert model.calls[0][0] == original_prompt          # attempt 0: original
    assert original_prompt in model.calls[1][0]          # attempt 1: repair (contains original)
    assert model.calls[1][0] != original_prompt          # attempt 1: but is different
    assert result.selected_candidate is not None
    assert len(result.attempts) == 3


def test_low_p_pass_uses_original_prompt(monkeypatch):
    """P(pass)=0.2 < threshold=0.5 → retry strategy: all attempts use original prompt."""
    original_prompt = make_task().prompt
    result, model, probe = _run(
        monkeypatch,
        generations=[_make_gen_with_hs(), _make_gen_no_hs(), _make_gen_no_hs()],
        executions=[
            make_execution_result(passed=False),
            make_execution_result(passed=False),
            make_execution_result(passed=True),
        ],
        p_pass=0.2,
        budget_B=3,
        threshold=0.5,
    )
    # All three attempts used the original prompt (best-of-B style).
    for prompt, _ in model.calls:
        assert prompt == original_prompt
    assert result.selected_candidate is not None
    assert len(result.attempts) == 3


def test_exactly_at_threshold_uses_repair(monkeypatch):
    """P(pass) == threshold is treated as >= threshold → repair."""
    original_prompt = make_task().prompt
    result, model, probe = _run(
        monkeypatch,
        generations=[_make_gen_with_hs(), _make_gen_no_hs()],
        executions=[
            make_execution_result(passed=False, stderr="err"),
            make_execution_result(passed=True),
        ],
        p_pass=0.5,
        budget_B=2,
        threshold=0.5,
    )
    # Attempt 1 should be a repair prompt (not original).
    assert model.calls[1][0] != original_prompt
    assert original_prompt in model.calls[1][0]


# ── Tests: budget exhaustion ──────────────────────────────────────────────────


def test_budget_exhausted_returns_none(monkeypatch):
    """If all B attempts fail, selected_candidate is None."""
    budget_B = 3
    result, model, probe = _run(
        monkeypatch,
        generations=[_make_gen_with_hs()] + [_make_gen_no_hs()] * (budget_B - 1),
        executions=[make_execution_result(passed=False)] * budget_B,
        p_pass=0.8,
        budget_B=budget_B,
    )
    assert result.selected_candidate is None
    assert len(result.attempts) == budget_B
    assert len(model.calls) == budget_B


# ── Tests: hidden states returned only when requested ────────────────────────


def test_hidden_states_not_returned_by_default(monkeypatch):
    """return_hidden_states=False → PolicyResult.hidden_states is None."""
    model = FakeModel([_make_gen_with_hs()])
    probe = FakeProbe(0.8)
    monkeypatch.setattr(
        "reasoninglab.policies.probe_guided.execute_candidate",
        lambda **_kw: make_execution_result(passed=True),
    )
    result = run_probe_guided(
        task=make_task(),
        model=model,
        budget_B=1,
        verifier_timeout_s=1.0,
        return_hidden_states=False,  # explicitly False
        probe=probe,
        probe_layer=_PROBE_LAYER,
        threshold=0.5,
    )
    assert result.hidden_states is None
