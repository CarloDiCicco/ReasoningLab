from __future__ import annotations

import re

from reasoninglab.eval.metrics import AttemptRecord
from reasoninglab.policies.contracts import PolicyResult, SupportsGenerate
from reasoninglab.tasks.schema import TaskRecord
from reasoninglab.verify.executor import execute_candidate
from reasoninglab.verify.taxonomy import classify_result


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


def run_baseline(
    task: TaskRecord,
    model: SupportsGenerate,
    budget_B: int,
    verifier_timeout_s: float | None = None,
    return_hidden_states: bool = False,
) -> PolicyResult:
    """Run the one-shot baseline policy for a single task."""
    if budget_B < 1:
        raise ValueError("budget_B must be >= 1")

    generation = model.generate(task.prompt, return_hidden_states=return_hidden_states)
    candidate_code = _extract_candidate_code(generation.text)
    
    '''
    Explicit override wins; otherwise use the per-task timeout contract. The line means:
    - If caller gives an explicit experiment override, use it.
    - Otherwise, trust each task's own timeout contract.
    '''
    timeout_s = verifier_timeout_s if verifier_timeout_s is not None else task.timeout_s

    execution = execute_candidate(
        candidate_code=candidate_code,
        test_code=task.test_code,
        timeout_s=timeout_s,
    )
    failure_type = classify_result(execution)

    # Attempt cost is model-generation cost plus verifier execution cost.
    attempt = AttemptRecord(
        task_id=task.task_id,
        policy="baseline",
        attempt_idx=0,
        passed=execution.passed,
        failure_type=failure_type,
        elapsed_s=generation.elapsed_s + execution.elapsed_s,
        tokens=generation.prompt_tokens + generation.completion_tokens,
    )

    return PolicyResult(
        attempts=(attempt,),
        selected_candidate=candidate_code if execution.passed else None,
    )
