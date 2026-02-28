"""Unit tests for _build_repair_prompt in isolation.

These verify the prompt template independently from any policy run,
similar to how _extract_candidate_code has its own tests in test_baseline.
"""

from __future__ import annotations

from reasoninglab.policies._utils import _MAX_STDERR_CHARS, _build_repair_prompt


def test_build_repair_prompt_contains_original_prompt():
    prompt = _build_repair_prompt(
        original_prompt="Write a function that adds two numbers.",
        candidate_code="def add(a, b): return a - b",
        stderr="AssertionError",
        failure_type="assertion",
    )
    assert "Write a function that adds two numbers." in prompt


def test_build_repair_prompt_contains_candidate_code():
    code = "def solve(x):\n    return x * 2"
    prompt = _build_repair_prompt(
        original_prompt="task",
        candidate_code=code,
        stderr="AssertionError",
        failure_type="assertion",
    )
    assert code in prompt


def test_build_repair_prompt_contains_stderr():
    stderr = "Traceback (most recent call last):\n  File ...\nNameError: name 'foo' is not defined"
    prompt = _build_repair_prompt(
        original_prompt="task",
        candidate_code="foo()",
        stderr=stderr,
        failure_type="runtime",
    )
    assert "NameError: name 'foo' is not defined" in prompt


def test_build_repair_prompt_contains_failure_type():
    prompt = _build_repair_prompt(
        original_prompt="task",
        candidate_code="code",
        stderr="SyntaxError: invalid syntax",
        failure_type="syntax",
    )
    assert "syntax" in prompt


def test_build_repair_prompt_truncates_long_stderr():
    # Build stderr that exceeds the limit.
    long_stderr = "x" * (_MAX_STDERR_CHARS + 500)
    prompt = _build_repair_prompt(
        original_prompt="task",
        candidate_code="code",
        stderr=long_stderr,
        failure_type="runtime",
    )
    # The full stderr should NOT appear (it was truncated).
    assert long_stderr not in prompt
    # The tail of the stderr (last _MAX_STDERR_CHARS chars) should be present.
    assert long_stderr[-_MAX_STDERR_CHARS:] in prompt
    # The truncation preamble should be present.
    assert f"showing the last {_MAX_STDERR_CHARS} chars of the stderr:" in prompt
