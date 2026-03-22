from __future__ import annotations

import numpy as np

from reasoninglab.eval.metrics import AttemptRecord
from reasoninglab.policies._utils import _build_repair_prompt, _extract_candidate_code
from reasoninglab.policies.contracts import PolicyResult, SupportsGenerate
from reasoninglab.tasks.schema import TaskRecord
from reasoninglab.verify.executor import execute_candidate
from reasoninglab.verify.taxonomy import classify_result


def run_probe_guided(
    task: TaskRecord,
    model: SupportsGenerate,
    budget_B: int,
    verifier_timeout_s: float | None = None,
    return_hidden_states: bool = False,
    *,
    probe,           # fitted sklearn Pipeline loaded from .joblib
    probe_layer: int = 35,
    threshold: float = 0.5,
) -> PolicyResult:
    """Run the probe-guided policy for a single task.

    Uses a linear probe trained on the model's hidden states to decide
    which retry strategy to use when attempt 0 fails:

      P(pass) >= threshold  →  REPAIR strategy (same as repair_b):
          Each subsequent attempt feeds the previous code + verifier
          error back to the model so it can fix its own mistake.

      P(pass) < threshold   →  RETRY strategy (same as best_of_b):
          Each subsequent attempt uses the original prompt from scratch.
          Stochastic sampling (do_sample=True) ensures diversity.

    The probe is called ONCE per task. Hidden states come from the prompt
    forward pass, which is deterministic — same prompt always produces the
    same hidden states regardless of sampling. Re-probing mid-task would
    return the same score, so one call is sufficient.

    The routing decision is fixed for the ENTIRE budget. This makes the
    policy predictable for any value of budget_B and keeps results easy
    to interpret: for each task, exactly one strategy is applied throughout.

    Args:
        task:                 The coding task to solve.
        model:                Model implementing SupportsGenerate.
        budget_B:             Maximum number of generation attempts.
        verifier_timeout_s:   Override for per-attempt verifier timeout.
        return_hidden_states: Whether to include hidden states in the result
                              (for downstream analysis). Note: the policy
                              always captures hidden states for attempt 0
                              internally to run the probe; this flag controls
                              what is returned in PolicyResult.
        probe:                Fitted sklearn Pipeline (StandardScaler → PCA
                              → LogisticRegression) loaded from a .joblib file.
        probe_layer:          Which transformer layer's hidden states to
                              extract and feed to the probe (default: 35,
                              the second-to-last decoder block of Qwen3-4B,
                              which consistently shows the strongest signal).
        threshold:            P(pass) cut-off between repair and retry.
                              Default 0.5 maps to the probe's natural decision
                              boundary (same as argmax classification).
    """
    if budget_B < 1:
        raise ValueError("budget_B must be >= 1")

    # Resolve verifier timeout once — task and override never change between attempts.
    timeout_s = verifier_timeout_s if verifier_timeout_s is not None else task.timeout_s

    attempts: list[AttemptRecord] = []
    # Hidden states are collected per attempt if return_hidden_states is requested.
    hidden_states_list: list[dict] = []
    selected_candidate: str | None = None

    # ── Attempt 0: probe the prompt and make the first generation ────────────
    # We always request hidden states for attempt 0 so the probe can score it.
    # Whether we return them in PolicyResult is controlled by return_hidden_states.
    generation = model.generate(task.prompt, return_hidden_states=True)

    # ── Score the prompt with the probe ──────────────────────────────────────
    # Extract the hidden state vector for the configured probe layer (default 35,
    # the second-to-last decoder block of Qwen3-4B). probe_layer is parametric —
    # any saved layer index can be used by changing the config.
    # Shape: (hidden_dim,) → reshape to (1, hidden_dim) for sklearn predict_proba.
    if generation.hidden_states is None or probe_layer not in generation.hidden_states:
        raise RuntimeError(
            f"Probe layer {probe_layer} not found in hidden states. "
            f"Available layers: {list(generation.hidden_states or {})}"
        )
    layer_vec = generation.hidden_states[probe_layer]  # shape: (hidden_dim,)
    p_pass = float(probe.predict_proba(layer_vec.reshape(1, -1))[0, 1])

    # Decide strategy ONCE based on the probe score. This decision holds for
    # all remaining attempts — no mid-task switching.
    # "repair" means: feed error feedback to the model so it can fix the code.
    # "retry"  means: fresh independent sample from the same prompt.
    use_repair = p_pass >= threshold
    strategy = "repair" if use_repair else "retry"

    # Log the probe score and routing decision for traceability.
    print(
        f"    [probe] P(pass)={p_pass:.3f} threshold={threshold} "
        f"→ {strategy}",
        flush=True,
    )

    # ── Execute attempt 0 ────────────────────────────────────────────────────
    candidate_code = _extract_candidate_code(generation.text)
    execution = execute_candidate(
        candidate_code=candidate_code,
        test_code=task.test_code,
        timeout_s=timeout_s,
    )
    failure_type = classify_result(execution)

    attempt = AttemptRecord(
        task_id=task.task_id,
        policy="probe_guided",
        attempt_idx=0,
        passed=execution.passed,
        failure_type=failure_type,
        elapsed_s=generation.elapsed_s + execution.elapsed_s,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        p_pass=p_pass,
        strategy=strategy,
    )
    attempts.append(attempt)

    # Collect attempt 0's hidden states if the caller requested them.
    # This block is structurally identical to the one inside the loop below —
    # both are needed because attempt 0 is handled outside the loop.
    if return_hidden_states and generation.hidden_states is not None:
        hidden_states_list.append(generation.hidden_states)

    if execution.passed:
        selected_candidate = candidate_code
        return PolicyResult(
            attempts=tuple(attempts),
            selected_candidate=selected_candidate,
            hidden_states=tuple(hidden_states_list) if hidden_states_list else None,
        )

    # ── Attempts 1 .. budget_B-1: follow the chosen strategy ─────────────────
    # Carry forward failure context for repair prompts (only used in repair path).
    last_code: str = candidate_code
    last_stderr: str | None = execution.stderr
    last_failure_type_value: str = failure_type.value

    for attempt_idx in range(1, budget_B):

        if use_repair:
            # REPAIR: build a prompt that includes the previous code + error feedback.
            # The model sees what went wrong and can iteratively correct its code.
            # Each iteration chains off the MOST RECENT failure (not cumulative).
            prompt = _build_repair_prompt(
                original_prompt=task.prompt,
                candidate_code=last_code,
                stderr=last_stderr,
                failure_type=last_failure_type_value,
            )
        else:
            # RETRY: reuse the original prompt for a fresh independent sample.
            # Stochastic sampling (do_sample=True, T=0.7) ensures each retry
            # produces a different output even though the prompt is identical.
            prompt = task.prompt

        # Hidden states not needed for attempts 1+: the probe already ran on attempt 0.
        generation = model.generate(prompt, return_hidden_states=return_hidden_states)

        candidate_code = _extract_candidate_code(generation.text)
        execution = execute_candidate(
            candidate_code=candidate_code,
            test_code=task.test_code,
            timeout_s=timeout_s,
        )
        failure_type = classify_result(execution)

        attempt = AttemptRecord(
            task_id=task.task_id,
            policy="probe_guided",
            attempt_idx=attempt_idx,
            passed=execution.passed,
            failure_type=failure_type,
            elapsed_s=generation.elapsed_s + execution.elapsed_s,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            p_pass=p_pass,
            strategy=strategy,
        )
        attempts.append(attempt)

        if return_hidden_states and generation.hidden_states is not None:
            hidden_states_list.append(generation.hidden_states)

        if execution.passed:
            selected_candidate = candidate_code
            break

        # Update failure context for the next repair iteration (no-op for retry path).
        last_code = candidate_code
        last_stderr = execution.stderr
        last_failure_type_value = failure_type.value

    return PolicyResult(
        attempts=tuple(attempts),
        selected_candidate=selected_candidate,
        hidden_states=tuple(hidden_states_list) if hidden_states_list else None,
    )
