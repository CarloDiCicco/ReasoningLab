from __future__ import annotations

import re


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_FENCED_BLOCK_RE = re.compile(
    r"```(?P<lang>[^\n`]*)\n(?P<code>.*?)```",
    re.DOTALL,
)

'''
Priority:
 first fenced block tagged python (case-insensitive),
 otherwise first fenced block of any type,
 otherwise raw text
'''
def _extract_candidate_code(raw_text: str) -> str:
    """Extract code from markdown fences, falling back to raw text."""
    # Strip any <think>...</think> reasoning content so fenced code blocks
    # inside thinking are not mistakenly selected over the actual answer.
    raw_text = _THINK_RE.sub("", raw_text)
    matches = list(_FENCED_BLOCK_RE.finditer(raw_text))
    if not matches:
        return raw_text

    # Prefer a python fence anywhere in the output; otherwise use first fence.
    for match in matches:
        language = match.group("lang").strip().lower()
        if language == "python":
            return match.group("code").strip()

    return matches[0].group("code").strip()


# ── Repair prompt builder ─────────────────────────────────────────────────────

# Maximum stderr characters included in repair prompts.
# Long tracebacks (recursive errors, large test suites) can blow up the model's
# context window. We keep the tail (not the head) because Python tracebacks put
# the actual error message at the end.
_MAX_STDERR_CHARS = 3000


def _build_repair_prompt(
    original_prompt: str,
    candidate_code: str,
    stderr: str,
    failure_type: str,
) -> str:
    """Build a repair prompt that includes verifier feedback from the previous attempt.

    The template is deliberately minimal — no chain-of-thought instructions or
    role preambles. This is a controlled H1 experiment: we measure the value of
    verifier feedback alone, not the value of prompt engineering.

    Args:
        original_prompt: The task description (same text the first attempt saw).
        candidate_code:  The code from the failed attempt.
        stderr:          Raw traceback / error output from the verifier.
        failure_type:    Normalized label as a plain string ("syntax", "assertion", etc.).
                         Passed as str (not FailureType enum) to keep this module
                         free of taxonomy imports.
    """
    # Truncate stderr from the front, keeping the tail where the actual error
    # message lives. Python tracebacks read bottom-up: the exception line is last.
    # A plain-English preamble tells the model it has a partial view so it can
    # reason accordingly rather than assuming the error is complete.
    if len(stderr) > _MAX_STDERR_CHARS:
        stderr = f"showing the last {_MAX_STDERR_CHARS} chars of the stderr:\n" + stderr[-_MAX_STDERR_CHARS:]

    return (
        f"{original_prompt}\n\n"
        f"Your previous attempt produced a {failure_type} error.\n\n"
        f"Previous code:\n"
        f"```python\n{candidate_code}\n```\n\n"
        f"Error output:\n"
        f"```\n{stderr}\n```\n\n"
        f"Fix the code and return a corrected version."
    )
