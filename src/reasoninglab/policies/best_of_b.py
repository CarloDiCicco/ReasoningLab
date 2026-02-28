from __future__ import annotations

from reasoninglab.eval.metrics import AttemptRecord
from reasoninglab.policies._utils import _extract_candidate_code
from reasoninglab.policies.contracts import PolicyResult, SupportsGenerate
from reasoninglab.tasks.schema import TaskRecord
from reasoninglab.verify.executor import execute_candidate
from reasoninglab.verify.taxonomy import classify_result


def run_best_of_b(
    task: TaskRecord,
    model: SupportsGenerate,
    budget_B: int,
    verifier_timeout_s: float | None = None,
    return_hidden_states: bool = False,
) -> PolicyResult:
    """Run the best-of-B policy for a single task.

    Makes up to B independent generation attempts (same prompt, i.i.d.).
    Stops early on the first passing attempt to avoid wasting budget.
    """
    if budget_B < 1:
        raise ValueError("budget_B must be >= 1")

    # Resolve timeout once here, outside the loop: the task and timeout never
    # change between attempts, so there is no reason to recompute per iteration.
    # Explicit caller override wins; otherwise fall back to the task's own contract.
    timeout_s = verifier_timeout_s if verifier_timeout_s is not None else task.timeout_s

    attempts: list[AttemptRecord] = []
    selected_candidate: str | None = None  # set only when an attempt passes

    for attempt_idx in range(budget_B):
        # Each call is i.i.d.: same prompt, fresh generation, no feedback from
        # previous attempts. That's what distinguishes best-of-B from repair-B.
        generation = model.generate(task.prompt, return_hidden_states=return_hidden_states)

        # Strip markdown fencing if the model wrapped its code in ``` blocks.
        candidate_code = _extract_candidate_code(generation.text)

        # Run the candidate against the task's test suite in an isolated subprocess.
        execution = execute_candidate(
            candidate_code=candidate_code,
            test_code=task.test_code,
            timeout_s=timeout_s,
        )

        # Map the raw ExecutionResult to a normalized FailureType label (or PASS).
        failure_type = classify_result(execution)

        # Cost = inference time + subprocess execution time; tokens = prompt + completion.
        attempt = AttemptRecord(
            task_id=task.task_id,
            policy="best_of_b",
            attempt_idx=attempt_idx,  # 0-based index within this task's budget
            passed=execution.passed,
            failure_type=failure_type,
            elapsed_s=generation.elapsed_s + execution.elapsed_s,
            tokens=generation.prompt_tokens + generation.completion_tokens,
        )
        attempts.append(attempt)

        # Early exit: no need to spend remaining budget once a passing solution is found.
        if execution.passed:
            selected_candidate = candidate_code
            break

    # Return ALL attempts made (not just the last), so the caller has the full
    # cost and failure-taxonomy data for this task regardless of outcome.
    return PolicyResult(
        attempts=tuple(attempts),
        selected_candidate=selected_candidate,  # None if all B attempts failed
    )
