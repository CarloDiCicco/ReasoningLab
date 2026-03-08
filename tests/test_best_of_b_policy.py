from __future__ import annotations

import pytest

# Import the module object so we can monkeypatch names inside it at test time.
import reasoninglab.policies.best_of_b as best_of_b_module
from reasoninglab.policies.best_of_b import run_best_of_b
from reasoninglab.verify.executor import ExecutionResult
from reasoninglab.verify.taxonomy import FailureType

from conftest import FakeGeneration, FakeModel, make_execution_result, make_gen, make_task


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_with_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    generations: list[FakeGeneration],
    executions: list[ExecutionResult],
    classifications: list[FailureType],
    verifier_timeout_s: float | None = None,
    task=None,
    budget_B: int = 5,
    return_hidden_states: bool = False,
):
    """Run best-of-B with monkeypatched executor and classifier.

    executions and classifications are consumed in order, one per loop iteration.
    captured_calls records the exact arguments passed to execute_candidate on
    each call, letting tests assert on wiring (timeout, test_code, etc.).

    Returns (model, result, captured_calls).
    """
    captured_calls: list[dict[str, object]] = []
    exec_index = 0

    def _fake_execute(candidate_code: str, test_code: str, timeout_s: float) -> ExecutionResult:
        nonlocal exec_index
        # Record what the policy actually passed to the executor this iteration.
        captured_calls.append({
            "candidate_code": candidate_code,
            "test_code": test_code,
            "timeout_s": timeout_s,
        })
        result = executions[exec_index]
        exec_index += 1
        return result

    classify_index = 0

    def _fake_classify(_result: ExecutionResult) -> FailureType:
        nonlocal classify_index
        ft = classifications[classify_index]
        classify_index += 1
        return ft

    # Patch on best_of_b_module (not on executor/taxonomy directly) because
    # run_best_of_b looks up these names in its own module namespace after import.
    monkeypatch.setattr(best_of_b_module, "execute_candidate", _fake_execute)
    monkeypatch.setattr(best_of_b_module, "classify_result", _fake_classify)

    model = FakeModel(generations)
    result = run_best_of_b(
        task=task or make_task(),
        model=model,
        budget_B=budget_B,
        verifier_timeout_s=verifier_timeout_s,
        return_hidden_states=return_hidden_states,
    )
    return model, result, captured_calls


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_best_of_b_exhausts_budget_when_all_fail(monkeypatch: pytest.MonkeyPatch):
    # When every attempt fails the policy must use the full budget, no early exit.
    budget = 3
    model, result, _ = _run_with_stubs(
        monkeypatch,
        generations=[make_gen() for _ in range(budget)],
        executions=[make_execution_result(passed=False)] * budget,
        classifications=[FailureType.ASSERTION] * budget,
        budget_B=budget,
    )
    assert len(model.calls) == budget
    assert len(result.attempts) == budget


def test_best_of_b_early_exits_on_first_pass(monkeypatch: pytest.MonkeyPatch):
    """Pass on attempt index 1 (second attempt) out of budget 5."""
    # Only 2 generations are provided; IndexError would fire if the policy
    # kept looping past the passing attempt — acts as a correctness guard.
    model, result, _ = _run_with_stubs(
        monkeypatch,
        generations=[make_gen(), make_gen(text="print('pass')")],
        executions=[make_execution_result(passed=False), make_execution_result(passed=True)],
        classifications=[FailureType.ASSERTION, FailureType.PASS],
        budget_B=5,
    )
    assert len(model.calls) == 2
    assert len(result.attempts) == 2


def test_best_of_b_single_attempt_pass_on_first_try(monkeypatch: pytest.MonkeyPatch):
    # Budget=5, but passes immediately: only 1 call, 1 record, candidate set.
    model, result, _ = _run_with_stubs(
        monkeypatch,
        generations=[make_gen(text="def solve(): return 1")],
        executions=[make_execution_result(passed=True)],
        classifications=[FailureType.PASS],
        budget_B=5,
    )
    assert len(model.calls) == 1
    assert len(result.attempts) == 1
    assert result.selected_candidate is not None


@pytest.mark.parametrize("budget", [0, -1])
def test_best_of_b_rejects_invalid_budget(budget: int):
    model = FakeModel([make_gen()])
    with pytest.raises(ValueError):
        run_best_of_b(task=make_task(), model=model, budget_B=budget)


def test_best_of_b_attempt_idx_is_sequential(monkeypatch: pytest.MonkeyPatch):
    # attempt_idx must be the 0-based position within the budget, in order.
    budget = 4
    _, result, _ = _run_with_stubs(
        monkeypatch,
        generations=[make_gen() for _ in range(budget)],
        executions=[make_execution_result(passed=False)] * budget,
        classifications=[FailureType.RUNTIME] * budget,
        budget_B=budget,
    )
    assert [a.attempt_idx for a in result.attempts] == list(range(budget))


def test_best_of_b_policy_field_on_all_records(monkeypatch: pytest.MonkeyPatch):
    # Every AttemptRecord must be labelled "best_of_b" regardless of outcome.
    budget = 3
    _, result, _ = _run_with_stubs(
        monkeypatch,
        generations=[make_gen() for _ in range(budget)],
        executions=[make_execution_result(passed=False)] * budget,
        classifications=[FailureType.ASSERTION] * budget,
        budget_B=budget,
    )
    assert all(a.policy == "best_of_b" for a in result.attempts)


def test_best_of_b_selected_candidate_on_pass(monkeypatch: pytest.MonkeyPatch):
    # selected_candidate must equal the extracted code that was actually passed
    # to the executor on the passing attempt (captured_calls[1]).
    _, result, captured = _run_with_stubs(
        monkeypatch,
        generations=[make_gen(), make_gen(text="def solve(): return 42")],
        executions=[make_execution_result(passed=False), make_execution_result(passed=True)],
        classifications=[FailureType.ASSERTION, FailureType.PASS],
        budget_B=5,
    )
    assert result.selected_candidate == captured[1]["candidate_code"]


def test_best_of_b_selected_candidate_none_on_all_fail(monkeypatch: pytest.MonkeyPatch):
    budget = 3
    _, result, _ = _run_with_stubs(
        monkeypatch,
        generations=[make_gen() for _ in range(budget)],
        executions=[make_execution_result(passed=False)] * budget,
        classifications=[FailureType.RUNTIME] * budget,
        budget_B=budget,
    )
    assert result.selected_candidate is None


def test_best_of_b_tokens_and_elapsed_per_attempt(monkeypatch: pytest.MonkeyPatch):
    # Each AttemptRecord must reflect its own generation + execution costs,
    # not a running total or a shared value.
    gens = [
        make_gen(prompt_tokens=10, completion_tokens=20, elapsed_s=0.1),
        make_gen(prompt_tokens=12, completion_tokens=34, elapsed_s=0.25),
    ]
    execs = [
        make_execution_result(passed=False, elapsed_s=0.4),
        make_execution_result(passed=False, elapsed_s=0.75),
    ]
    _, result, _ = _run_with_stubs(
        monkeypatch,
        generations=gens,
        executions=execs,
        classifications=[FailureType.ASSERTION, FailureType.RUNTIME],
        budget_B=2,
    )
    assert result.attempts[0].prompt_tokens == 10
    assert result.attempts[0].completion_tokens == 20
    assert result.attempts[0].elapsed_s == pytest.approx(0.5)   # 0.1 + 0.4
    assert result.attempts[1].prompt_tokens == 12
    assert result.attempts[1].completion_tokens == 34
    assert result.attempts[1].elapsed_s == pytest.approx(1.0)   # 0.25 + 0.75


def test_best_of_b_uses_verifier_timeout_override(monkeypatch: pytest.MonkeyPatch):
    # When an explicit override is given, every execute call must receive it.
    budget = 2
    _, _, captured = _run_with_stubs(
        monkeypatch,
        generations=[make_gen() for _ in range(budget)],
        executions=[make_execution_result(passed=False)] * budget,
        classifications=[FailureType.SYNTAX] * budget,
        verifier_timeout_s=1.75,
        budget_B=budget,
    )
    assert all(c["timeout_s"] == 1.75 for c in captured)


def test_best_of_b_falls_back_to_task_timeout(monkeypatch: pytest.MonkeyPatch):
    # When no override is given, every execute call must use task.timeout_s.
    budget = 2
    task = make_task()
    _, _, captured = _run_with_stubs(
        monkeypatch,
        generations=[make_gen() for _ in range(budget)],
        executions=[make_execution_result(passed=False)] * budget,
        classifications=[FailureType.SYNTAX] * budget,
        task=task,
        budget_B=budget,
    )
    assert all(c["timeout_s"] == task.timeout_s for c in captured)


@pytest.mark.parametrize("return_hidden_states", [False, True])
def test_best_of_b_forwards_return_hidden_states(
    monkeypatch: pytest.MonkeyPatch, return_hidden_states: bool
):
    # The flag must be forwarded on every generate() call, not just the first.
    budget = 2
    model, _, _ = _run_with_stubs(
        monkeypatch,
        generations=[make_gen() for _ in range(budget)],
        executions=[make_execution_result(passed=False)] * budget,
        classifications=[FailureType.RUNTIME] * budget,
        budget_B=budget,
        return_hidden_states=return_hidden_states,
    )
    assert all(call[1] is return_hidden_states for call in model.calls)


def test_best_of_b_passes_prompt_unchanged_on_every_call(monkeypatch: pytest.MonkeyPatch):
    # The policy must pass the prompt verbatim on every attempt — no trimming
    # or reformatting, even across multiple iterations.
    prompt = "  Keep spacing.\nAnd new lines.\n"
    task = make_task(prompt=prompt)
    budget = 3
    model, _, _ = _run_with_stubs(
        monkeypatch,
        generations=[make_gen() for _ in range(budget)],
        executions=[make_execution_result(passed=False)] * budget,
        classifications=[FailureType.RUNTIME] * budget,
        task=task,
        budget_B=budget,
    )
    assert all(call[0] == task.prompt for call in model.calls)
