from __future__ import annotations

import re


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
    matches = list(_FENCED_BLOCK_RE.finditer(raw_text))
    if not matches:
        return raw_text

    # Prefer a python fence anywhere in the output; otherwise use first fence.
    for match in matches:
        language = match.group("lang").strip().lower()
        if language == "python":
            return match.group("code").strip()

    return matches[0].group("code").strip()
