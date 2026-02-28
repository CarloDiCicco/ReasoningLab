"""
tests/test_taxonomy.py
──────────────────────
Unit tests for verify/taxonomy.py.

Tests cover:
  - FailureType enum values and str/JSON serialization
  - classify_result() for each FailureType branch
  - Priority ordering (PASS first, TIMEOUT before stderr parsing, SYNTAX before ASSERTION)
  - Edge cases (empty stderr, mixed markers, whitespace-only stderr)
"""

from __future__ import annotations

import json

import pytest

from reasoninglab.verify.executor import ExecutionResult
from reasoninglab.verify.taxonomy import FailureType, classify_result


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

# _result is an helper internal function used by other methods inside this file to run tests
def _result(
    passed: bool = False,
    timed_out: bool = False,
    stdout: str = "",
    stderr: str = "",
    elapsed_s: float = 1.0,
    exit_code: int | None = 1,
) -> ExecutionResult:
    """Construct an ExecutionResult with sensible defaults for test brevity."""
    return ExecutionResult(
        passed=passed,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
        elapsed_s=elapsed_s,
        exit_code=exit_code,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FailureType enum
# ─────────────────────────────────────────────────────────────────────────────

class TestFailureTypeEnum:
    """Verify enum member values and serialization properties."""

    def test_values(self):
        """Each member's value must match the expected plain string."""
        assert FailureType.PASS.value      == "pass"
        assert FailureType.TIMEOUT.value   == "timeout"
        assert FailureType.SYNTAX.value    == "syntax"
        assert FailureType.ASSERTION.value == "assertion"
        assert FailureType.RUNTIME.value   == "runtime"
        assert FailureType.UNKNOWN.value   == "unknown"

    def test_equality_with_string(self):
        """str-Enum members must compare equal to their string value.
        This is relied on by JSONL filtering code that compares against
        string literals without importing FailureType."""
        for ft in FailureType:
            assert ft == ft.value

    def test_json_serialization(self):
        """json.dumps() must serialize each member as the plain string value,
        not as a Python repr.  The JSONL logger depends on this."""
        for ft in FailureType:
            assert json.dumps(ft) == json.dumps(ft.value)

    def test_all_members_present(self):
        """Guard against accidental removal of a member."""
        expected = {"pass", "timeout", "syntax", "assertion", "runtime", "unknown"}
        assert {ft.value for ft in FailureType} == expected


# ─────────────────────────────────────────────────────────────────────────────
# classify_result — happy paths
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyResultBranches:
    """One test per FailureType branch; each is the canonical / cleanest case."""

    def test_pass(self):
        r = _result(passed=True, exit_code=0)
        assert classify_result(r) == FailureType.PASS

    def test_timeout(self):
        r = _result(timed_out=True, exit_code=None)
        assert classify_result(r) == FailureType.TIMEOUT

    def test_syntax_error(self):
        stderr = (
            '  File "<string>", line 1\n'
            "    def foo(\n"
            "           ^\n"
            "SyntaxError: invalid syntax\n"
        )
        assert classify_result(_result(stderr=stderr)) == FailureType.SYNTAX

    def test_indentation_error(self):
        stderr = "IndentationError: unexpected indent\n"
        assert classify_result(_result(stderr=stderr)) == FailureType.SYNTAX

    def test_tab_error(self):
        stderr = "TabError: inconsistent use of tabs and spaces in indentation\n"
        assert classify_result(_result(stderr=stderr)) == FailureType.SYNTAX

    def test_assertion_error_bare(self):
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "<string>", line 5, in <module>\n'
            "    assert result == 42\n"
            "AssertionError\n"
        )
        assert classify_result(_result(stderr=stderr)) == FailureType.ASSERTION

    def test_assertion_error_with_message(self):
        stderr = "AssertionError: expected 3, got 5\n"
        assert classify_result(_result(stderr=stderr)) == FailureType.ASSERTION

    def test_runtime_name_error(self):
        stderr = "NameError: name 'solution' is not defined\n"
        assert classify_result(_result(stderr=stderr)) == FailureType.RUNTIME

    def test_runtime_type_error(self):
        stderr = "TypeError: unsupported operand type(s) for +: 'int' and 'str'\n"
        assert classify_result(_result(stderr=stderr)) == FailureType.RUNTIME

    def test_runtime_recursion_error(self):
        stderr = "RecursionError: maximum recursion depth exceeded\n"
        assert classify_result(_result(stderr=stderr)) == FailureType.RUNTIME

    def test_runtime_value_error(self):
        stderr = "ValueError: math domain error\n"
        assert classify_result(_result(stderr=stderr)) == FailureType.RUNTIME

    def test_unknown_empty_stderr(self):
        # passed=False, timed_out=False, stderr="" → hard crash or OOM
        assert classify_result(_result(exit_code=1)) == FailureType.UNKNOWN

    def test_unknown_whitespace_only_stderr(self):
        # Whitespace-only stderr should be treated the same as empty.
        assert classify_result(_result(stderr="   \n  \t\n")) == FailureType.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# classify_result — priority ordering
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyResultPriority:
    """Verify that earlier checks win over later ones when conditions overlap."""

    def test_pass_beats_timed_out(self):
        # Logically impossible (executor never sets both to True), but the
        # function should still honour the PASS-first rule.
        r = _result(passed=True, timed_out=True, exit_code=0)
        assert classify_result(r) == FailureType.PASS

    def test_pass_beats_nonempty_stderr(self):
        # A passing run that printed something to stderr (e.g. a DeprecationWarning)
        # must still be classified as PASS.
        r = _result(passed=True, stderr="DeprecationWarning: ...\n", exit_code=0)
        assert classify_result(r) == FailureType.PASS

    def test_timeout_beats_stderr(self):
        # If for some reason stderr is non-empty on a timeout (partial output
        # before the kill), TIMEOUT still wins.
        r = _result(timed_out=True, stderr="SyntaxError: ...\n", exit_code=None)
        assert classify_result(r) == FailureType.TIMEOUT

    def test_syntax_beats_assertion(self):
        # Logically impossible: a SyntaxError prevents execution, so
        # AssertionError cannot appear in the same run.  But if both strings
        # were somehow present, SYNTAX must win (more fundamental failure).
        stderr = "SyntaxError: bad syntax\nAssertionError\n"
        assert classify_result(_result(stderr=stderr)) == FailureType.SYNTAX

    def test_assertion_beats_runtime(self):
        # A stderr that contains both AssertionError and a RuntimeError-style
        # message — AssertionError wins because we check it first.
        stderr = "AssertionError\nRuntimeError: something\n"
        assert classify_result(_result(stderr=stderr)) == FailureType.ASSERTION
