from __future__ import annotations

import pytest

# Import the module object so we can monkeypatch names inside it at test time.
import reasoninglab.policies.repair_b as repair_b_module
from reasoninglab.policies._utils import _MAX_STDERR_CHARS
from reasoninglab.policies.repair_b import run_repair_b
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
    """Run repair-B with monkeypatched executor and classifier.

    Same list-based pattern as best-of-B tests. captured_calls records
    each execute_candidate invocation.

    Returns (model, result, captured_calls).
    """
    captured_calls: list[dict[str, object]] = []
    exec_index = 0

    def _fake_execute(candidate_code: str, test_code: str, timeout_s: float) -> ExecutionResult:
        nonlocal exec_index
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

    # Patch on repair_b_module (not on executor/taxonomy directly) because
    # run_repair_b looks up these names in its own module namespace after import.
    monkeypatch.setattr(repair_b_module, "execute_candidate", _fake_execute)
    monkeypatch.setattr(repair_b_module, "classify_result", _fake_classify)

    model = FakeModel(generations)
    result = run_repair_b(
        task=task or make_task(),
        model=model,
        budget_B=budget_B,
        verifier_timeout_s=verifier_timeout_s,
        return_hidden_states=return_hidden_states,
    )
    return model, result, captured_calls


# ── Tests: prompt construction (the core differentiator from best-of-B) ──────

def test_repair_b_first_attempt_uses_original_prompt(monkeypatch: pytest.MonkeyPatch):
    # The first attempt must use the task prompt verbatim, just like baseline.
    task = make_task(prompt="Write a function that adds two numbers.")
    model, _, _ = _run_with_stubs(
        monkeypatch,
        generations=[make_gen()],
        executions=[make_execution_result(passed=True)],
        classifications=[FailureType.PASS],
        task=task,
        budget_B=1,
    )
    assert model.calls[0][0] == task.prompt


def test_repair_b_subsequent_attempts_use_repair_prompt(monkeypatch: pytest.MonkeyPatch):
    # After a failure, the next prompt must include the original task, the failed
    # code, the stderr, and the failure type — not just the original prompt.
    task = make_task(prompt="Solve the problem.")
    stderr_text = "NameError: name 'foo' is not defined"
    model, _, _ = _run_with_stubs(
        monkeypatch,
        generations=[make_gen(text="foo()"), make_gen(text="bar()")],
        executions=[
            make_execution_result(passed=False, stderr=stderr_text),
            make_execution_result(passed=True),
        ],
        classifications=[FailureType.RUNTIME, FailureType.PASS],
        task=task,
        budget_B=5,
    )
    repair_prompt = model.calls[1][0]
    # The repair prompt must contain all four pieces of feedback.
    assert task.prompt in repair_prompt
    assert "foo()" in repair_prompt
    assert stderr_text in repair_prompt
    assert "runtime" in repair_prompt


def test_repair_b_repair_prompt_includes_stderr_and_failure_type(monkeypatch: pytest.MonkeyPatch):
    # Verify that different failure types and stderr content are reflected correctly.
    stderr_text = "SyntaxError: invalid syntax (line 1)"
    model, _, _ = _run_with_stubs(
        monkeypatch,
        generations=[make_gen(text="def("), make_gen(text="def f(): pass")],
        executions=[
            make_execution_result(passed=False, stderr=stderr_text),
            make_execution_result(passed=True),
        ],
        classifications=[FailureType.SYNTAX, FailureType.PASS],
        budget_B=5,
    )
    repair_prompt = model.calls[1][0]
    assert stderr_text in repair_prompt
    assert "syntax" in repair_prompt


def test_repair_b_repair_prompt_truncates_long_stderr(monkeypatch: pytest.MonkeyPatch):
    # Very long stderr must be truncated to _MAX_STDERR_CHARS (keeping the tail).
    long_stderr = "x" * (_MAX_STDERR_CHARS + 500)
    model, _, _ = _run_with_stubs(
        monkeypatch,
        generations=[make_gen(), make_gen()],
        executions=[
            make_execution_result(passed=False, stderr=long_stderr),
            make_execution_result(passed=False),
        ],
        classifications=[FailureType.RUNTIME, FailureType.RUNTIME],
        budget_B=2,
    )
    repair_prompt = model.calls[1][0]
    # The full long stderr should NOT appear.
    assert long_stderr not in repair_prompt
    # The tail should be present.
    assert long_stderr[-_MAX_STDERR_CHARS:] in repair_prompt


# ── Tests: loop behavior ─────────────────────────────────────────────────────

def test_repair_b_early_exits_on_first_pass(monkeypatch: pytest.MonkeyPatch):
    # Pass on the very first attempt — only 1 call, no repair needed.
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


def test_repair_b_early_exits_on_repair_pass(monkeypatch: pytest.MonkeyPatch):
    # Fail attempt 0, pass attempt 1 — exactly 2 calls.
    model, result, _ = _run_with_stubs(
        monkeypatch,
        generations=[make_gen(), make_gen(text="def solve(): return 42")],
        executions=[
            make_execution_result(passed=False, stderr="AssertionError"),
            make_execution_result(passed=True),
        ],
        classifications=[FailureType.ASSERTION, FailureType.PASS],
        budget_B=5,
    )
    assert len(model.calls) == 2
    assert len(result.attempts) == 2
    assert result.selected_candidate is not None


def test_repair_b_exhausts_budget_when_all_fail(monkeypatch: pytest.MonkeyPatch):
    budget = 3
    model, result, _ = _run_with_stubs(
        monkeypatch,
        generations=[make_gen() for _ in range(budget)],
        executions=[make_execution_result(passed=False, stderr="err")] * budget,
        classifications=[FailureType.RUNTIME] * budget,
        budget_B=budget,
    )
    assert len(model.calls) == budget
    assert len(result.attempts) == budget


@pytest.mark.parametrize("budget", [0, -1])
def test_repair_b_rejects_invalid_budget(budget: int):
    model = FakeModel([make_gen()])
    with pytest.raises(ValueError):
        run_repair_b(task=make_task(), model=model, budget_B=budget)


# ── Tests: result structure ──────────────────────────────────────────────────

def test_repair_b_selected_candidate_set_on_pass(monkeypatch: pytest.MonkeyPatch):
    _, result, captured = _run_with_stubs(
        monkeypatch,
        generations=[make_gen(), make_gen(text="def solve(): return 42")],
        executions=[
            make_execution_result(passed=False, stderr="err"),
            make_execution_result(passed=True),
        ],
        classifications=[FailureType.ASSERTION, FailureType.PASS],
        budget_B=5,
    )
    assert result.selected_candidate == captured[1]["candidate_code"]


def test_repair_b_selected_candidate_none_on_all_fail(monkeypatch: pytest.MonkeyPatch):
    budget = 3
    _, result, _ = _run_with_stubs(
        monkeypatch,
        generations=[make_gen() for _ in range(budget)],
        executions=[make_execution_result(passed=False, stderr="err")] * budget,
        classifications=[FailureType.RUNTIME] * budget,
        budget_B=budget,
    )
    assert result.selected_candidate is None


def test_repair_b_policy_field_on_all_records(monkeypatch: pytest.MonkeyPatch):
    budget = 3
    _, result, _ = _run_with_stubs(
        monkeypatch,
        generations=[make_gen() for _ in range(budget)],
        executions=[make_execution_result(passed=False, stderr="err")] * budget,
        classifications=[FailureType.ASSERTION] * budget,
        budget_B=budget,
    )
    assert all(a.policy == "repair_b" for a in result.attempts)


def test_repair_b_tokens_and_elapsed_per_attempt(monkeypatch: pytest.MonkeyPatch):
    # Each AttemptRecord must reflect its own generation + execution costs.
    gens = [
        make_gen(prompt_tokens=10, completion_tokens=20, elapsed_s=0.1),
        make_gen(prompt_tokens=12, completion_tokens=34, elapsed_s=0.25),
    ]
    execs = [
        make_execution_result(passed=False, elapsed_s=0.4, stderr="err"),
        make_execution_result(passed=False, elapsed_s=0.75),
    ]
    _, result, _ = _run_with_stubs(
        monkeypatch,
        generations=gens,
        executions=execs,
        classifications=[FailureType.ASSERTION, FailureType.RUNTIME],
        budget_B=2,
    )
    assert result.attempts[0].tokens == 30          # 10 + 20
    assert result.attempts[0].elapsed_s == pytest.approx(0.5)   # 0.1 + 0.4
    assert result.attempts[1].tokens == 46          # 12 + 34
    assert result.attempts[1].elapsed_s == pytest.approx(1.0)   # 0.25 + 0.75


# ── Tests: repair feedback provenance ────────────────────────────────────────

def test_repair_b_uses_most_recent_failure_not_first(monkeypatch: pytest.MonkeyPatch):
    # With 3 attempts (fail A → fail B → pass), the repair prompt for attempt 2
    # must contain attempt 1's code and stderr (fail B), NOT attempt 0's (fail A).
    # This verifies that last_code/last_stderr are overwritten each iteration,
    # not fixed to the first failure.
    code_A = "def attempt_zero(): pass"
    code_B = "def attempt_one(): pass"
    stderr_A = "AssertionError: attempt zero failed"
    stderr_B = "AssertionError: attempt one failed"

    model, _, _ = _run_with_stubs(
        monkeypatch,
        generations=[
            make_gen(text=code_A),
            make_gen(text=code_B),
            make_gen(text="def solve(): return 42"),
        ],
        executions=[
            make_execution_result(passed=False, stderr=stderr_A),
            make_execution_result(passed=False, stderr=stderr_B),
            make_execution_result(passed=True),
        ],
        classifications=[FailureType.ASSERTION, FailureType.ASSERTION, FailureType.PASS],
        budget_B=5,
    )
    repair_prompt_attempt_2 = model.calls[2][0]
    # Must reference attempt 1's artifacts.
    assert code_B in repair_prompt_attempt_2
    assert stderr_B in repair_prompt_attempt_2
    # Must NOT reference attempt 0's artifacts.
    assert code_A not in repair_prompt_attempt_2
    assert stderr_A not in repair_prompt_attempt_2


# ── Tests: timeout and hidden states wiring ──────────────────────────────────

def test_repair_b_uses_verifier_timeout_override(monkeypatch: pytest.MonkeyPatch):
    budget = 2
    _, _, captured = _run_with_stubs(
        monkeypatch,
        generations=[make_gen() for _ in range(budget)],
        executions=[make_execution_result(passed=False, stderr="err")] * budget,
        classifications=[FailureType.SYNTAX] * budget,
        verifier_timeout_s=1.75,
        budget_B=budget,
    )
    assert all(c["timeout_s"] == 1.75 for c in captured)


def test_repair_b_falls_back_to_task_timeout(monkeypatch: pytest.MonkeyPatch):
    budget = 2
    task = make_task()
    _, _, captured = _run_with_stubs(
        monkeypatch,
        generations=[make_gen() for _ in range(budget)],
        executions=[make_execution_result(passed=False, stderr="err")] * budget,
        classifications=[FailureType.SYNTAX] * budget,
        task=task,
        budget_B=budget,
    )
    assert all(c["timeout_s"] == task.timeout_s for c in captured)


@pytest.mark.parametrize("return_hidden_states", [False, True])
def test_repair_b_forwards_return_hidden_states(
    monkeypatch: pytest.MonkeyPatch, return_hidden_states: bool
):
    budget = 2
    model, _, _ = _run_with_stubs(
        monkeypatch,
        generations=[make_gen() for _ in range(budget)],
        executions=[make_execution_result(passed=False, stderr="err")] * budget,
        classifications=[FailureType.RUNTIME] * budget,
        budget_B=budget,
        return_hidden_states=return_hidden_states,
    )
    assert all(call[1] is return_hidden_states for call in model.calls)
