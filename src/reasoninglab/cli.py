"""
cli.py
──────
CLI entrypoint for ReasoningLab H1 experiments.

Loads a YAML/JSON config, optionally overrides fields from the command line,
then runs one policy arm across all tasks.

Usage:
    python -m reasoninglab.cli --config configs/h1/smoke.yaml
    python -m reasoninglab.cli --config configs/h1/smoke.yaml --policy best_of_b
    python -m reasoninglab.cli --config configs/h1/smoke.yaml --budget 5 --max-tasks 10

Flow:
    1. Load ExperimentConfig from --config path
    2. Apply any CLI overrides (--policy, --budget, --max-tasks)
    3. Load tasks via loaders.load_tasks()
    4. Load model via QwenModel()
    5. Wrap policy_fn with progress printing
    6. Call run_experiment() → RunMetrics
    7. Print summary table

Design choices:
    - Progress printing lives here, not in run_experiment(). The CLI wraps
      policy_fn in a closure that prints [K/N] task_id before each call.
      This avoids adding a progress_callback parameter to the runner API.
    - Policy dispatch is a simple dict lookup, not a plugin system. H1 has
      exactly three fixed policies; cli.py is not a general-purpose tool.
    - Exceptions from the runner propagate unchanged. Fail fast.
"""

from __future__ import annotations

import argparse
import functools
from datetime import datetime
from pathlib import Path

import joblib

from reasoninglab.config.schema import load_experiment_config
from reasoninglab.eval.metrics import RunMetrics
from reasoninglab.eval.runner import run_experiment
from reasoninglab.models.qwen_local import QwenModel
from reasoninglab.policies.baseline import run_baseline
from reasoninglab.policies.best_of_b import run_best_of_b
from reasoninglab.policies.contracts import PolicyResult
from reasoninglab.policies.probe_guided import run_probe_guided
from reasoninglab.policies.repair_b import run_repair_b
from reasoninglab.tasks.loaders import load_tasks


# ── Policy dispatch table ─────────────────────────────────────────────────────
# Maps config.policy string to the actual callable.
# probe_guided is bound with its probe and thresholds via functools.partial in main().

_POLICY_FN = {
    "baseline": run_baseline,
    "best_of_b": run_best_of_b,
    "repair_b": run_repair_b,
    "probe_guided": run_probe_guided,
}


# ── CLI argument parsing ──────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m reasoninglab.cli",
        description="Run a ReasoningLab H1 experiment.",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to ExperimentConfig YAML/JSON file.",
    )
    parser.add_argument(
        "--policy", choices=list(_POLICY_FN),
        help="Override config.policy.",
    )
    parser.add_argument(
        "--budget", type=int,
        help="Override config.budget_B.",
    )
    parser.add_argument(
        "--max-tasks", type=int, dest="max_tasks",
        help="Override config.max_tasks (cap number of tasks to evaluate).",
    )
    return parser.parse_args()


# ── Summary printing ──────────────────────────────────────────────────────────

def _print_summary(policy_name: str, metrics: RunMetrics, run_dir: Path) -> None:
    """Print a formatted summary table to stdout after the run completes."""
    print()
    print("── Run summary ────────────────────────────────────────────────")
    print(f"  policy        : {policy_name}")
    print(f"  tasks         : {metrics.total_tasks}")
    print(f"  attempts      : {metrics.total_attempts}")
    print(f"  pass_rate     : {metrics.pass_rate:.1%}")
    print(f"  mean_prompt_tokens    : {metrics.mean_prompt_tokens:.1f}")
    print(f"  mean_completion_tokens: {metrics.mean_completion_tokens:.1f}")
    print(f"  mean_elapsed_s: {metrics.mean_elapsed_s:.3f}s")
    if metrics.failure_dist:
        print("  failure_dist  :")
        for ft, frac in sorted(metrics.failure_dist.items(), key=lambda kv: -kv[1]):
            print(f"    {ft:<12} {frac:.1%}")
    print(f"  artifacts     : {run_dir}")
    print("───────────────────────────────────────────────────────────────")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── Load config with optional CLI overrides ───────────────────────────────
    config = load_experiment_config(args.config)

    overrides: dict = {}
    if args.policy is not None:
        overrides["policy"] = args.policy
    if args.budget is not None:
        overrides["budget_B"] = args.budget
    if args.max_tasks is not None:
        overrides["max_tasks"] = args.max_tasks
    if overrides:
        config = config.model_copy(update=overrides)

    policy_name: str = config.policy
    policy_fn_base = _POLICY_FN[policy_name]

    # ── Probe-guided: load the probe and bind it to the policy function ───────
    # For probe_guided, we load the serialized sklearn Pipeline from disk and
    # bind it (along with probe_layer and threshold) as keyword-only defaults
    # using functools.partial. This keeps the policy function's public signature
    # identical to all other policies (task, model, budget_B, ...) from the
    # runner's perspective, while injecting the probe transparently.
    if policy_name == "probe_guided":
        if config.probe is None:
            raise ValueError(
                "Policy 'probe_guided' requires a [probe] section in the config "
                "with probe_path, probe_layer, and threshold."
            )
        probe_path = Path(config.probe.probe_path)
        if not probe_path.exists():
            raise FileNotFoundError(f"Probe file not found: {probe_path}")
        print(f"[cli] Loading probe from {probe_path} ...", flush=True)
        probe_pipeline = joblib.load(probe_path)
        policy_fn_base = functools.partial(
            run_probe_guided,
            probe=probe_pipeline,
            probe_layer=config.probe.probe_layer,
            threshold=config.probe.threshold,
        )
        print(
            f"[cli] Probe loaded: layer={config.probe.probe_layer}, "
            f"threshold={config.probe.threshold}",
            flush=True,
        )

    # ── Load tasks ────────────────────────────────────────────────────────────
    print(f"[cli] Loading tasks from {config.paths.tasks_file} ...", flush=True)
    tasks = load_tasks(config.paths.tasks_file, limit=config.max_tasks)
    n = len(tasks)
    print(f"[cli] Loaded {n} tasks.", flush=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[cli] Loading model: {config.model_id} ...", flush=True)
    model = QwenModel(config.model_id, config.generation)
    if config.runtime.capture_device_map:
        print(f"[cli] hf_device_map: {model.hf_device_map}", flush=True)

    # ── Build output paths ────────────────────────────────────────────────────
    # Timestamp makes each run unique so re-running doesn't overwrite results.
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.paths.runs_root / f"{config.experiment_id}_{run_stamp}"
    attempts_path = run_dir / "attempts.jsonl"
    summary_path = run_dir / "summary.jsonl"
    print(f"[cli] Writing artifacts to {run_dir}", flush=True)

    # ── Wrap policy with progress printing ────────────────────────────────────
    # The CLI wraps policy_fn in a closure that prints progress before each task.
    # This keeps progress reporting in the CLI layer without adding a callback
    # parameter to run_experiment().
    call_idx = 0

    def _policy_with_progress(**kwargs) -> PolicyResult:
        nonlocal call_idx
        call_idx += 1
        task_id = kwargs["task"].task_id
        print(f"[cli] [{call_idx}/{n}] {task_id}", flush=True)
        return policy_fn_base(**kwargs)

    # ── Run experiment ────────────────────────────────────────────────────────
    print(
        f"[cli] Starting run: policy={policy_name}, budget_B={config.budget_B}",
        flush=True,
    )
    # H2: hidden states directory sits alongside attempts.jsonl in the run dir.
    hidden_states_dir = run_dir / "hidden_states" if config.runtime.return_hidden_states else None

    metrics = run_experiment(
        tasks=tasks,
        model=model,
        policy_fn=_policy_with_progress,
        policy_name=policy_name,
        budget_B=config.budget_B,
        attempts_path=attempts_path,
        summary_path=summary_path,
        verifier_timeout_s=config.runtime.verifier_timeout_s,
        return_hidden_states=config.runtime.return_hidden_states,
        hidden_states_dir=hidden_states_dir,
        extra_summary_fields={
            "model_id":           config.model_id,
            "tasks_file":         str(config.paths.tasks_file),
            "budget_B":           config.budget_B,
            "verifier_timeout_s": config.runtime.verifier_timeout_s,
            "max_new_tokens":     config.generation.max_new_tokens,
            "do_sample":          config.generation.do_sample,
            "temperature":        config.generation.temperature,
            "top_p":              config.generation.top_p,
            "top_k":              config.generation.top_k,
            "min_p":              config.generation.min_p,
            "repetition_penalty": config.generation.repetition_penalty,
            "presence_penalty":   config.generation.presence_penalty,
        },
    )

    _print_summary(policy_name, metrics, run_dir)


if __name__ == "__main__":
    main()
