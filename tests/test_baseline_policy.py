from __future__ import annotations

from dataclasses import dataclass

import pytest

import reasoninglab.policies.baseline as baseline_module
from reasoninglab.policies.baseline import _extract_candidate_code, run_baseline
from reasoninglab.tasks.schema import TaskRecord
from reasoninglab.verify.executor import ExecutionResult
from reasoninglab.verify.taxonomy import FailureType


@dataclass
class _FakeGeneration:
    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float


class _FakeModel:
    def __init__(self, generation: _FakeGeneration) -> None:
        self._generation = generation
        self.calls: list[tuple[str, bool]] = []

    def generate(self, prompt: str, return_hidden_states: bool = False) -> _FakeGeneration:
        self.calls.append((prompt, return_hidden_states))
        return self._generation


def _task(prompt: str = "Solve this.\nReturn valid Python.") -> TaskRecord:
    return TaskRecord(
        task_id="task-1",
        prompt=prompt,
        test_code="assert True",
        timeout_s=8.0,
    )


def _execution_result(passed: bool, elapsed_s: float = 0.5) -> ExecutionResult:
    return ExecutionResult(
        passed=passed,
        timed_out=False,
        stdout="",
        stderr="",
        elapsed_s=elapsed_s,
        exit_code=0 if passed else 1,
    )


def _run_with_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    generation: _FakeGeneration,
    execution: ExecutionResult,
    classified_as: FailureType,
    verifier_timeout_s: float | None = None,
    task: TaskRecord | None = None,
    budget_B: int = 1,
):
    captured: dict[str, object] = {}

    def _fake_execute(candidate_code: str, test_code: str, timeout_s: float) -> ExecutionResult:
        captured["candidate_code"] = candidate_code
        captured["test_code"] = test_code
        captured["timeout_s"] = timeout_s
        return execution

    monkeypatch.setattr(baseline_module, "execute_candidate", _fake_execute)
    monkeypatch.setattr(baseline_module, "classify_result", lambda _: classified_as)

    model = _FakeModel(generation)
    result = run_baseline(
        task=task or _task(),
        model=model,
        budget_B=budget_B,
        verifier_timeout_s=verifier_timeout_s,
    )
    return model, result, captured


def test_baseline_calls_generate_once_even_when_budget_is_larger(monkeypatch: pytest.MonkeyPatch):
    model, _, _ = _run_with_stubs(
        monkeypatch,
        generation=_FakeGeneration("print('x')", 4, 3, 0.2),
        execution=_execution_result(passed=False),
        classified_as=FailureType.ASSERTION,
        budget_B=5,
    )
    assert len(model.calls) == 1


def test_baseline_uses_task_prompt_unchanged(monkeypatch: pytest.MonkeyPatch):
    prompt = "  Keep spacing.\nAnd new lines.\n"
    task = _task(prompt=prompt)
    model, _, _ = _run_with_stubs(
        monkeypatch,
        generation=_FakeGeneration("print('x')", 4, 3, 0.2),
        execution=_execution_result(passed=False),
        classified_as=FailureType.RUNTIME,
        task=task,
    )
    assert model.calls[0][0] == task.prompt


@pytest.mark.parametrize("return_hidden_states", [False, True])
def test_baseline_forwards_return_hidden_states(
    monkeypatch: pytest.MonkeyPatch, return_hidden_states: bool
):
    model = _FakeModel(_FakeGeneration("print('x')", 4, 3, 0.2))
    '''
    monkeypatching replace the "execute_candidate" and "classify_result" functions
    with dumb stubs that return a fixed ExecutionResult and FailureType respectively
    '''
    monkeypatch.setattr(baseline_module, "execute_candidate", lambda *_, **__: _execution_result(passed=False))
    monkeypatch.setattr(baseline_module, "classify_result", lambda _: FailureType.RUNTIME)
    run_baseline(task=_task(), model=model, budget_B=1, return_hidden_states=return_hidden_states)
    assert model.calls[0][1] is return_hidden_states


def test_baseline_tracks_tokens_and_elapsed_as_generation_plus_execution(
    monkeypatch: pytest.MonkeyPatch,
):
    _, result, _ = _run_with_stubs(
        monkeypatch,
        generation=_FakeGeneration("print('x')", 12, 34, 0.25),
        execution=_execution_result(passed=False, elapsed_s=0.75),
        classified_as=FailureType.ASSERTION,
    )
    record = result.attempts[0]
    assert record.tokens == 46
    assert record.elapsed_s == pytest.approx(1.0)


def test_baseline_uses_verifier_timeout_override_when_provided(monkeypatch: pytest.MonkeyPatch):
    _, _, captured = _run_with_stubs(
        monkeypatch,
        generation=_FakeGeneration("print('x')", 4, 3, 0.2),
        execution=_execution_result(passed=False),
        classified_as=FailureType.SYNTAX,
        verifier_timeout_s=1.75,
    )
    assert captured["timeout_s"] == 1.75


def test_baseline_falls_back_to_task_timeout_when_override_missing(monkeypatch: pytest.MonkeyPatch):
    task = _task()
    _, _, captured = _run_with_stubs(
        monkeypatch,
        generation=_FakeGeneration("print('x')", 4, 3, 0.2),
        execution=_execution_result(passed=False),
        classified_as=FailureType.SYNTAX,
        task=task,
    )
    assert captured["timeout_s"] == task.timeout_s


def test_baseline_uses_classifier_output_for_failure_type(monkeypatch: pytest.MonkeyPatch):
    _, result, _ = _run_with_stubs(
        monkeypatch,
        generation=_FakeGeneration("print('x')", 4, 3, 0.2),
        execution=_execution_result(passed=False),
        classified_as=FailureType.RUNTIME,
    )
    assert result.attempts[0].failure_type == FailureType.RUNTIME


def test_baseline_returns_single_attempt_tuple_and_selected_candidate_on_pass(
    monkeypatch: pytest.MonkeyPatch,
):
    _, result, captured = _run_with_stubs(
        monkeypatch,
        generation=_FakeGeneration("print('passed')", 6, 5, 0.3),
        execution=_execution_result(passed=True),
        classified_as=FailureType.PASS,
    )
    assert isinstance(result.attempts, tuple)
    assert len(result.attempts) == 1
    assert result.attempts[0].policy == "baseline"
    assert result.attempts[0].attempt_idx == 0
    assert result.selected_candidate == captured["candidate_code"]


@pytest.mark.parametrize("budget", [0, -1])
def test_baseline_rejects_invalid_budget(
    budget: int,
):
    model = _FakeModel(_FakeGeneration("print('x')", 4, 3, 0.2))
    with pytest.raises(ValueError):
        run_baseline(
            task=_task(),
            model=model,
            budget_B=budget,
        )


def test_extract_candidate_code_prefers_first_python_fence():
    text = (
        "Intro\n"
        "```text\n"
        "not_python()\n"
        "```\n"
        "```PYTHON\n"
        "def solve():\n"
        "    return 1\n"
        "```\n"
    )
    assert _extract_candidate_code(text) == "def solve():\n    return 1"


def test_extract_candidate_code_uses_first_generic_fence_when_no_python():
    text = (
        "```javascript\n"
        "console.log('first')\n"
        "```\n"
        "```sql\n"
        "select 1;\n"
        "```\n"
    )
    assert _extract_candidate_code(text) == "console.log('first')"


def test_extract_candidate_code_falls_back_to_raw_text():
    raw = "def solve(x):\n    return x + 1\n"
    assert _extract_candidate_code(raw) == raw
