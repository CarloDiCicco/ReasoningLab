from __future__ import annotations

import pytest

# Import the module object so we can monkeypatch names inside it at test time.
import reasoninglab.policies.baseline as baseline_module
from reasoninglab.policies._utils import _extract_candidate_code
from reasoninglab.policies.baseline import run_baseline
from reasoninglab.verify.executor import ExecutionResult
from reasoninglab.verify.taxonomy import FailureType

from conftest import FakeGeneration, FakeModel, make_execution_result, make_gen, make_task


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_with_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    generation: FakeGeneration,
    execution: ExecutionResult,
    classified_as: FailureType,
    verifier_timeout_s: float | None = None,
    task=None,
    budget_B: int = 1,
):
    # captured holds the exact arguments that run_baseline passed to execute_candidate,
    # letting tests assert on wiring (e.g. correct timeout, correct test_code).
    captured: dict[str, object] = {}

    def _fake_execute(candidate_code: str, test_code: str, timeout_s: float) -> ExecutionResult:
        captured["candidate_code"] = candidate_code
        captured["test_code"] = test_code
        captured["timeout_s"] = timeout_s
        return execution

    # Patch on baseline_module (not on executor/taxonomy directly) because
    # run_baseline looks up these names in its own module namespace after import.
    monkeypatch.setattr(baseline_module, "execute_candidate", _fake_execute)
    monkeypatch.setattr(baseline_module, "classify_result", lambda _: classified_as)

    # Baseline only makes one call, so wrap the single generation in a list.
    model = FakeModel([generation])
    result = run_baseline(
        task=task or make_task(),
        model=model,
        budget_B=budget_B,
        verifier_timeout_s=verifier_timeout_s,
    )
    return model, result, captured


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_baseline_calls_generate_once_even_when_budget_is_larger(monkeypatch: pytest.MonkeyPatch):
    # Baseline is one-shot by design: budget_B is accepted but ignored after the
    # guard check. A larger budget must not cause extra generate() calls.
    model, _, _ = _run_with_stubs(
        monkeypatch,
        generation=FakeGeneration("print('x')", 4, 3, 0.2),
        execution=make_execution_result(passed=False),
        classified_as=FailureType.ASSERTION,
        budget_B=5,
    )
    assert len(model.calls) == 1


def test_baseline_uses_task_prompt_unchanged(monkeypatch: pytest.MonkeyPatch):
    # The policy must pass the prompt verbatim — no trimming or reformatting.
    prompt = "  Keep spacing.\nAnd new lines.\n"
    task = make_task(prompt=prompt)
    model, _, _ = _run_with_stubs(
        monkeypatch,
        generation=FakeGeneration("print('x')", 4, 3, 0.2),
        execution=make_execution_result(passed=False),
        classified_as=FailureType.RUNTIME,
        task=task,
    )
    assert model.calls[0][0] == task.prompt


@pytest.mark.parametrize("return_hidden_states", [False, True])
def test_baseline_forwards_return_hidden_states(
    monkeypatch: pytest.MonkeyPatch, return_hidden_states: bool
):
    model = FakeModel([FakeGeneration("print('x')", 4, 3, 0.2)])
    # Minimal stubs — we only care that generate() received the right flag.
    monkeypatch.setattr(baseline_module, "execute_candidate", lambda *_, **__: make_execution_result(passed=False))
    monkeypatch.setattr(baseline_module, "classify_result", lambda _: FailureType.RUNTIME)
    run_baseline(task=make_task(), model=model, budget_B=1, return_hidden_states=return_hidden_states)
    assert model.calls[0][1] is return_hidden_states


def test_baseline_tracks_tokens_and_elapsed_as_generation_plus_execution(
    monkeypatch: pytest.MonkeyPatch,
):
    # AttemptRecord.tokens and elapsed_s must be the sum of inference + execution costs.
    _, result, _ = _run_with_stubs(
        monkeypatch,
        generation=FakeGeneration("print('x')", 12, 34, 0.25),
        execution=make_execution_result(passed=False, elapsed_s=0.75),
        classified_as=FailureType.ASSERTION,
    )
    record = result.attempts[0]
    assert record.tokens == 46        # 12 + 34
    assert record.elapsed_s == pytest.approx(1.0)  # 0.25 + 0.75


def test_baseline_uses_verifier_timeout_override_when_provided(monkeypatch: pytest.MonkeyPatch):
    _, _, captured = _run_with_stubs(
        monkeypatch,
        generation=FakeGeneration("print('x')", 4, 3, 0.2),
        execution=make_execution_result(passed=False),
        classified_as=FailureType.SYNTAX,
        verifier_timeout_s=1.75,
    )
    assert captured["timeout_s"] == 1.75


def test_baseline_falls_back_to_task_timeout_when_override_missing(monkeypatch: pytest.MonkeyPatch):
    task = make_task()
    _, _, captured = _run_with_stubs(
        monkeypatch,
        generation=FakeGeneration("print('x')", 4, 3, 0.2),
        execution=make_execution_result(passed=False),
        classified_as=FailureType.SYNTAX,
        task=task,
    )
    assert captured["timeout_s"] == task.timeout_s


def test_baseline_uses_classifier_output_for_failure_type(monkeypatch: pytest.MonkeyPatch):
    # The policy must store whatever classify_result returns, not re-derive it.
    _, result, _ = _run_with_stubs(
        monkeypatch,
        generation=FakeGeneration("print('x')", 4, 3, 0.2),
        execution=make_execution_result(passed=False),
        classified_as=FailureType.RUNTIME,
    )
    assert result.attempts[0].failure_type == FailureType.RUNTIME


def test_baseline_returns_single_attempt_tuple_and_selected_candidate_on_pass(
    monkeypatch: pytest.MonkeyPatch,
):
    _, result, captured = _run_with_stubs(
        monkeypatch,
        generation=FakeGeneration("print('passed')", 6, 5, 0.3),
        execution=make_execution_result(passed=True),
        classified_as=FailureType.PASS,
    )
    assert isinstance(result.attempts, tuple)
    assert len(result.attempts) == 1
    assert result.attempts[0].policy == "baseline"
    assert result.attempts[0].attempt_idx == 0
    # selected_candidate must equal the extracted code that was passed to the executor.
    assert result.selected_candidate == captured["candidate_code"]


@pytest.mark.parametrize("budget", [0, -1])
def test_baseline_rejects_invalid_budget(budget: int):
    model = FakeModel([FakeGeneration("print('x')", 4, 3, 0.2)])
    with pytest.raises(ValueError):
        run_baseline(task=make_task(), model=model, budget_B=budget)


# ── _extract_candidate_code unit tests ────────────────────────────────────────
# These test the parsing logic in isolation, independent of any policy run.

def test_extract_candidate_code_prefers_first_python_fence():
    text = (
        "Intro\n"
        "```text\n"
        "not_python()\n"
        "```\n"
        "```PYTHON\n"          # case-insensitive match
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
    # No fences at all — return the raw text unchanged.
    raw = "def solve(x):\n    return x + 1\n"
    assert _extract_candidate_code(raw) == raw
