"""Budget accounting tests: no policy may ever exceed budget_B model calls per task.

These are regression guards — if any policy's loop condition is wrong (e.g., an
off-by-one in range(budget_B), or a missing break), this file catches it.

Strategy:
  - Parametrize over all three policies and multiple budget values.
  - Provide exactly budget_B failing generations so FakeModel raises IndexError
    if the policy tries to call generate() one extra time — this acts as a
    built-in over-call guard without requiring an explicit assertion.
  - Separately assert len(model.calls) == expected_calls to make the contract
    explicit and produce a readable failure message.

Monkeypatching targets each policy's own module namespace (the same pattern used
in the individual policy test files) because that is where execute_candidate and
classify_result are looked up at call time.
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


# ── Shared stub wiring ────────────────────────────────────────────────────────

def _patch_policy(monkeypatch: pytest.MonkeyPatch, module, *, all_fail: bool = True):
    """Monkeypatch execute_candidate and classify_result on the given policy module.

    When all_fail=True every execution result is a failure (so the policy never
    exits early on a pass), letting us measure the maximum number of calls used.
    """
    failure_type = FailureType.ASSERTION if all_fail else FailureType.PASS

    monkeypatch.setattr(
        module, "execute_candidate",
        lambda *_args, **_kwargs: make_execution_result(passed=not all_fail, stderr="err"),
    )
    monkeypatch.setattr(
        module, "classify_result",
        lambda _result: failure_type,
    )


# ── Parametrized budget-cap test ──────────────────────────────────────────────

# Each tuple: (run_fn, policy_module, expected_calls_when_all_fail).
# Baseline is one-shot by design: expected_calls is always 1 regardless of budget_B.
# Best-of-B and repair-B exhaust the full budget when all attempts fail.
_POLICIES = [
    (run_baseline,  baseline_module,  "one_shot"),
    (run_best_of_b, best_of_b_module, "full_budget"),
    (run_repair_b,  repair_b_module,  "full_budget"),
]

@pytest.mark.parametrize("policy_fn,policy_module,call_mode", _POLICIES)
@pytest.mark.parametrize("budget_B", [1, 3, 5])
def test_policy_never_exceeds_budget(
    monkeypatch: pytest.MonkeyPatch,
    policy_fn,
    policy_module,
    call_mode: str,
    budget_B: int,
):
    """No policy should call model.generate() more than budget_B times per task.

    Providing exactly budget_B generations to FakeModel also acts as a guard:
    any extra call raises IndexError and fails the test loudly.
    """
    _patch_policy(monkeypatch, policy_module, all_fail=True)

    # Provide as many generations as the maximum any policy could request.
    model = FakeModel([make_gen() for _ in range(budget_B)])

    policy_fn(task=make_task(), model=model, budget_B=budget_B)

    # Explicit upper-bound assertion for clarity.
    assert len(model.calls) <= budget_B


@pytest.mark.parametrize("policy_fn,policy_module,call_mode", _POLICIES)
@pytest.mark.parametrize("budget_B", [1, 3, 5])
def test_policy_exhausts_full_budget_when_all_fail(
    monkeypatch: pytest.MonkeyPatch,
    policy_fn,
    policy_module,
    call_mode: str,
    budget_B: int,
):
    """When every attempt fails, multi-attempt policies must use every slot.

    Baseline is one-shot: it always uses exactly 1 call regardless of budget_B.
    Best-of-B and repair-B must consume all budget_B calls (no premature exit
    when passing candidates haven't been found).
    """
    _patch_policy(monkeypatch, policy_module, all_fail=True)
    model = FakeModel([make_gen() for _ in range(budget_B)])

    policy_fn(task=make_task(), model=model, budget_B=budget_B)

    expected = 1 if call_mode == "one_shot" else budget_B
    assert len(model.calls) == expected


# ── Baseline-specific: always exactly one call ────────────────────────────────

@pytest.mark.parametrize("budget_B", [1, 5, 10])
def test_baseline_always_uses_exactly_one_call(
    monkeypatch: pytest.MonkeyPatch, budget_B: int
):
    """Baseline is one-shot by design: budget_B is accepted but the extra budget
    must never trigger additional generate() calls.
    """
    _patch_policy(monkeypatch, baseline_module, all_fail=True)
    # Provide more generations than baseline should ever use.
    model = FakeModel([make_gen() for _ in range(budget_B)])

    run_baseline(task=make_task(), model=model, budget_B=budget_B)

    assert len(model.calls) == 1


# ── Early-exit guard: multi-attempt policies stop as soon as a pass is found ──

@pytest.mark.parametrize("policy_fn,policy_module,call_mode", [
    (run_best_of_b, best_of_b_module, "full_budget"),
    (run_repair_b,  repair_b_module,  "full_budget"),
])
@pytest.mark.parametrize("pass_on_attempt", [0, 1, 2])
def test_multi_attempt_policy_exits_immediately_on_pass(
    monkeypatch: pytest.MonkeyPatch,
    policy_fn,
    policy_module,
    call_mode: str,
    pass_on_attempt: int,
):
    """Multi-attempt policies must stop as soon as one attempt passes, not
    continue consuming the remaining budget unnecessarily.
    """
    budget_B = 5
    # Alternate: fail on all attempts before pass_on_attempt, then pass.
    exec_results = [
        make_execution_result(passed=(i == pass_on_attempt), stderr="err" if i != pass_on_attempt else "")
        for i in range(budget_B)
    ]
    failure_types = [
        FailureType.PASS if i == pass_on_attempt else FailureType.ASSERTION
        for i in range(budget_B)
    ]

    exec_idx = 0
    classify_idx = 0

    def _fake_exec(*_args, **_kwargs):
        nonlocal exec_idx
        r = exec_results[exec_idx]
        exec_idx += 1
        return r

    def _fake_classify(_result):
        nonlocal classify_idx
        ft = failure_types[classify_idx]
        classify_idx += 1
        return ft

    monkeypatch.setattr(policy_module, "execute_candidate", _fake_exec)
    monkeypatch.setattr(policy_module, "classify_result", _fake_classify)

    # Provide only (pass_on_attempt + 1) generations — any extra call raises IndexError.
    model = FakeModel([make_gen() for _ in range(pass_on_attempt + 1)])

    policy_fn(task=make_task(), model=model, budget_B=budget_B)

    assert len(model.calls) == pass_on_attempt + 1
