from __future__ import annotations

from reasoninglab.eval.metrics import AttemptRecord
from reasoninglab.policies._utils import _extract_candidate_code
from reasoninglab.policies.contracts import PolicyResult, SupportsGenerate
from reasoninglab.tasks.schema import TaskRecord
from reasoninglab.verify.executor import execute_candidate
from reasoninglab.verify.taxonomy import classify_result


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
