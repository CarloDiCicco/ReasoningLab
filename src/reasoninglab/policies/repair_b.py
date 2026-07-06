from __future__ import annotations

from reasoninglab.eval.metrics import AttemptRecord
from reasoninglab.policies._utils import _build_repair_prompt, _extract_candidate_code
from reasoninglab.policies.contracts import PolicyResult, SupportsGenerate
from reasoninglab.tasks.schema import TaskRecord
from reasoninglab.verify.executor import execute_candidate
from reasoninglab.verify.taxonomy import classify_result


def run_repair_b(
    task: TaskRecord,
    model: SupportsGenerate,
    budget_B: int,
    verifier_timeout_s: float | None = None,
    return_hidden_states: bool = False,
) -> PolicyResult:
    """Run the repair-B policy for a single task.

    Makes 1 initial attempt with the original prompt, then up to (B-1) repair
    attempts where each prompt includes the previous attempt's code and verifier
    feedback (stderr + failure type). Stops early on the first passing attempt.

    This is the key difference from best-of-B: attempts are NOT independent.
    Each repair attempt is conditioned on the previous failure, so the model
    can iteratively correct its code rather than starting from scratch.
    """
    if budget_B < 1:
        raise ValueError("budget_B must be >= 1")

    # Resolve timeout once — the task and override never change between attempts.
    timeout_s = verifier_timeout_s if verifier_timeout_s is not None else task.timeout_s

    attempts: list[AttemptRecord] = []
    hidden_states_list: list[dict] = []  # one dict per attempt (H2)
    selected_candidate: str | None = None

    # State carried across iterations for building the repair prompt.
    # Initialized to None; only populated after a failed attempt.
    last_code: str | None = None
    last_stderr: str | None = None
    last_failure_type_value: str | None = None

    for attempt_idx in range(budget_B):
        # First attempt: use the original task prompt (no prior context).
        # Subsequent attempts: build a repair prompt with verifier feedback from
        # the most recent failure so the model can see what went wrong.
        if attempt_idx == 0:
            prompt = task.prompt
        else:
            prompt = _build_repair_prompt(
                original_prompt=task.prompt,
                candidate_code=last_code,  # type: ignore[arg-type]  # guaranteed non-None after attempt 0
                stderr=last_stderr,  # type: ignore[arg-type]
                failure_type=last_failure_type_value,  # type: ignore[arg-type]
            )

        generation = model.generate(prompt, return_hidden_states=return_hidden_states)

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
        # Persist the extracted candidate code (and a truncated stderr) so this run
        # can later be re-graded under a different test suite — this isolates the
        # effect of the test-suite fix from run-to-run generation stochasticity.
        attempt = AttemptRecord(
            task_id=task.task_id,
            policy="repair_b",
            attempt_idx=attempt_idx,
            passed=execution.passed,
            failure_type=failure_type,
            elapsed_s=generation.elapsed_s + execution.elapsed_s,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            code=candidate_code,
            stderr=(execution.stderr or "")[:4000] if execution.stderr else None,
        )
        attempts.append(attempt)
        if generation.hidden_states is not None:
            hidden_states_list.append(generation.hidden_states)

        # Early exit: no need to spend remaining budget once a passing solution is found.
        if execution.passed:
            selected_candidate = candidate_code
            break

        # Carry forward the failure context for the next repair iteration.
        # The repair prompt builder needs the code, stderr, and failure label
        # from the most recent (not cumulative) failure.
        last_code = candidate_code
        last_stderr = execution.stderr
        last_failure_type_value = failure_type.value

    # Return ALL attempts made so the caller has the full cost and failure-taxonomy
    # data for this task. The repair trajectory (attempt 0 → 1 → ... → N) is
    # also needed by H2's trajectory test (probe score monotonicity).
    return PolicyResult(
        attempts=tuple(attempts),
        selected_candidate=selected_candidate,
        hidden_states=tuple(hidden_states_list) if hidden_states_list else None,
    )
