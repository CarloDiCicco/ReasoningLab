"""Shared test helpers for policy tests.

These stand-ins replace real I/O boundaries (model inference, subprocess
execution) so tests run instantly and deterministically. Extracted here
because baseline, best-of-B, and repair-B tests all need the same fakes.

Each policy test file keeps its own _run_with_stubs() function since the
monkeypatch target and parameter shape differ per policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from reasoninglab.tasks.schema import TaskRecord
from reasoninglab.verify.executor import ExecutionResult


@dataclass
class FakeGeneration:
    """Mirrors the fields of GenerationLike that policy functions read."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float
    hidden_states: dict | None = None


class FakeModel:
    """Model stub that returns generations from a pre-set sequence.

    Yields one generation per call in order. IndexError on over-call is
    intentional: it loudly fails the test if the policy makes more calls
    than expected, acting as a built-in budget guard.
    """

    def __init__(self, generations: list[FakeGeneration]) -> None:
        self._generations = list(generations)
        self._index = 0
        # Each entry is (prompt, return_hidden_states) — inspected by tests.
        self.calls: list[tuple[str, bool]] = []

    def generate(self, prompt: str, return_hidden_states: bool = False) -> FakeGeneration:
        self.calls.append((prompt, return_hidden_states))
        gen = self._generations[self._index]
        self._index += 1
        return gen


def make_task(prompt: str = "Solve this.\nReturn valid Python.") -> TaskRecord:
    """Create a minimal TaskRecord for testing."""
    return TaskRecord(
        task_id="task-1",
        prompt=prompt,
        test_code="assert True",
        timeout_s=8.0,
    )


def make_execution_result(passed: bool, elapsed_s: float = 0.5, stderr: str = "") -> ExecutionResult:
    """Create an ExecutionResult with sensible defaults.

    stderr defaults to empty but can be set for repair-B tests that need
    verifier feedback content.
    """
    return ExecutionResult(
        passed=passed,
        timed_out=False,
        stdout="",
        stderr=stderr,
        elapsed_s=elapsed_s,
        exit_code=0 if passed else 1,
    )


def make_gen(
    text: str = "print('x')",
    prompt_tokens: int = 4,
    completion_tokens: int = 3,
    elapsed_s: float = 0.2,
) -> FakeGeneration:
    """Shorthand for a FakeGeneration with sensible defaults."""
    return FakeGeneration(text, prompt_tokens, completion_tokens, elapsed_s)
