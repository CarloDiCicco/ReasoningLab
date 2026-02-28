"""
verify/taxonomy.py
──────────────────
Maps a raw ExecutionResult to a normalized FailureType label.

Used by:
  - eval/metrics.py      : aggregates failure counts per policy arm (H1 analysis)
  - io/jsonl_logger.py   : writes the label as a plain string field in JSONL records
  - policies/repair_b.py : (future) adapts the repair prompt based on failure type

Classification logic (applied in priority order — first match wins):
  1. PASS      – result.passed is True (returncode==0 AND sentinel found)
  2. TIMEOUT   – result.timed_out is True (stderr empty; process was killed)
  3. SYNTAX    – SyntaxError / IndentationError / TabError in stderr
  4. ASSERTION – AssertionError in stderr (tests ran; logic is wrong)
  5. RUNTIME   – any other non-empty stderr (NameError, TypeError, etc.)
  6. UNKNOWN   – stderr is empty, but the attempt still failed (OOM / hard crash)

Why this ordering matters for H1:
  SYNTAX errors mean the model produced unparseable code — the fix strategy is
  different from ASSERTION errors (parseable code with wrong logic).  Separating
  these two is essential for the failure-mode distribution analysis that H1 uses
  to compare repair vs best-of behaviour.
"""

from __future__ import annotations

import json
from enum import Enum

from reasoninglab.verify.executor import ExecutionResult


# ─────────────────────────────────────────────────────────────────────────────
# FailureType
# ─────────────────────────────────────────────────────────────────────────────

# str and Enum are needed for the string recognition 
class FailureType(str, Enum):
    """Normalized failure category for a single code execution attempt.

    Inherits from str so each member IS the string value (e.g. FailureType.PASS
    == "pass").  Two practical consequences:

      1. json.dumps() serializes members as plain strings without a custom
         encoder — the JSON encoder tests isinstance(obj, str) first, which
         returns True for str-Enum members.

      2. Equality comparisons with raw strings work directly:
             result_dict["failure_type"] == "syntax"   # True if SYNTAX
         so callers can filter JSONL logs without importing this module.
    """

    PASS      = "pass"       # returncode == 0 AND sentinel found in stdout
    TIMEOUT   = "timeout"    # subprocess killed for exceeding wall-clock limit
    SYNTAX    = "syntax"     # SyntaxError / IndentationError / TabError
    ASSERTION = "assertion"  # AssertionError (tests ran; logic is wrong)
    RUNTIME   = "runtime"    # any other exception (NameError, TypeError, etc.)
    UNKNOWN   = "unknown"    # failure with no stderr (OOM / hard crash)


# ─────────────────────────────────────────────────────────────────────────────
# Internal constants
# ─────────────────────────────────────────────────────────────────────────────

# Exception class names that indicate a parse/compile failure.
# IndentationError and TabError are Python subclasses of SyntaxError, but they
# appear under their own names in tracebacks, so all three must be listed.
# A SyntaxError means the model's code never started executing — the interpreter
# rejected it at parse time.
_SYNTAX_MARKERS: tuple[str, ...] = ("SyntaxError", "IndentationError", "TabError")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def classify_result(result: ExecutionResult) -> FailureType:
    """Map a raw ExecutionResult to a FailureType label.

    The checks are applied in a fixed priority order; the first matching
    condition is returned.  This order is deliberate — see module docstring.

    Args:
        result: Structured output from execute_candidate().

    Returns:
        A FailureType enum member.  The member IS the string value, so callers
        can pass it directly to json.dumps() or compare with string literals.
    """

    # ── 1. Pass ──────────────────────────────────────────────────────────────
    # Definitive signal from the executor: both returncode==0 AND the sentinel
    # were present.  This check comes first because passed=True is authoritative
    # regardless of anything else in the result.
    if result.passed:
        return FailureType.PASS

    # ── 2. Timeout ───────────────────────────────────────────────────────────
    # The subprocess was killed by the wall-clock timeout.  stderr will be empty
    # (the OS kills the process, no Python exception is raised), so we must rely
    # on the explicit timed_out flag rather than stderr parsing.
    if result.timed_out:
        return FailureType.TIMEOUT

    # ── 3–6. Stderr-based classification ─────────────────────────────────────
    # At this point: passed=False, timed_out=False.
    # The process ran to completion (or crashed) without the sentinel appearing.
    # We inspect stderr to determine why.

    stderr = result.stderr

    # ── 3. Syntax ─────────────────────────────────────────────────────────────
    # SyntaxError / IndentationError / TabError mean the interpreter rejected
    # the model's code before any test ran.  Checked before ASSERTION so that
    # a malformed script is never misclassified as a logic error.
    if any(marker in stderr for marker in _SYNTAX_MARKERS):
        return FailureType.SYNTAX

    # ── 4. Assertion ──────────────────────────────────────────────────────────
    # AssertionError means the code was syntactically valid and ran, but a test
    # assertion failed.  This is the "logic error" category: the model understood
    # the problem structure but produced a wrong answer.
    if "AssertionError" in stderr:
        return FailureType.ASSERTION

    # ── 5. Runtime ────────────────────────────────────────────────────────────
    # Non-empty stderr that matched none of the patterns above.  Covers the long
    # tail of runtime exceptions: NameError, TypeError, RecursionError,
    # ValueError, ZeroDivisionError, AttributeError, etc.
    if stderr.strip():
        return FailureType.RUNTIME

    # ── 6. Unknown ────────────────────────────────────────────────────────────
    # stderr is empty but the attempt still failed (passed=False, timed_out=False).
    # Likely causes: OOM kill by the OS, segfault (faulthandler disabled so no
    # crash report), or a C-level exit that bypassed our preamble.  We cannot
    # say more without additional tooling — log it for manual inspection.
    return FailureType.UNKNOWN
