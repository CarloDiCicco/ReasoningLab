#!/usr/bin/env python3
"""Analyze hidden-state trajectories from multi-attempt repair_b runs.

Loads layer-35 hidden states across repair attempts and runs 5 analyses:
  1. Per-attempt pass/fail classifier (can the model's state predict success?)
  2. RepE-style trajectory direction (is there a "repair-success" direction?)
  3. Convergence analysis (do successful trajectories move consistently?)
  4. PCA trajectory visualization (2D projection of repair paths)
  5. Distance to success region (do repairs move toward the "pass" centroid?)

Requires: pip install -e ".[probe]"   (installs scikit-learn + matplotlib)

Usage:
    python scripts/analyze_trajectories.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — we save PNGs, no GUI needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from scipy import stats

from reasoninglab.probing.data import _task_id_from_filename
from reasoninglab.probing.probe import train_probe


# ── Configuration ────────────────────────────────────────────────────────────
# Hardcoded paths and parameters. Change these when running on different data.

RUN_DIR = Path("runs\h2-trajectory-all")
LAYER = 35          # Penultimate decoder block — best signal from prior probe analysis
OUTPUT_DIR = Path("results/h2/trajectory_analysis")
SEED = 0
TOKEN_LIMIT = 768   # max_new_tokens from config — used to detect repetition loops


# ── Data structures ─────────────────────────────────────────────────────────
# We define two simple dataclasses to hold trajectory data.
# These live in the script (not the library) because they are specific to
# this analysis and not reused elsewhere.

@dataclass
class AttemptInfo:
    """One repair attempt within a trajectory."""
    attempt_idx: int
    hidden_state: np.ndarray   # shape (hidden_dim,) — the layer-35 activation
                               # captured AFTER the model reads the repair prompt
                               # and BEFORE it generates the repair code
    passed: bool               # did the generated code pass all test cases?
    failure_type: str           # "pass", "syntax", "runtime", "assertion", "timeout"
    completion_tokens: int      # how many tokens the model generated
    prompt_tokens: int          # how many tokens in the prompt (input to the model)


@dataclass
class TrajectoryRecord:
    """A full repair trajectory for one coding task."""
    task_id: str
    attempts: list[AttemptInfo]  # sorted by attempt_idx (0, 1, 2, ...)
    eventually_passes: bool     # True if ANY attempt passed
    is_repetition_loop: bool    # True if the model is stuck generating the same
                                # broken output every time (same failure + token limit)


# ── Data loading ─────────────────────────────────────────────────────────────

# Regex to parse .npz keys like "attempt_2_layer_35".
# Each .npz file stores hidden states for ALL attempts of one task, with keys
# like "attempt_0_layer_29", "attempt_0_layer_35", "attempt_1_layer_35", etc.
# We only care about the layer specified in LAYER.
_ATTEMPT_LAYER_RE = re.compile(r"^attempt_(\d+)_layer_(\d+)$")


def load_trajectories(run_dir: Path, layer: int) -> list[TrajectoryRecord]:
    """Load multi-attempt hidden states and join with attempt metadata.

    Data comes from two sources that must be joined:
      1. attempts.jsonl — one JSON line per attempt, has metadata like
         failure_type and completion_tokens but NO hidden states
      2. hidden_states/*.npz — one file per task, has hidden-state vectors
         for all attempts and layers but NO metadata

    We join them by (task_id, attempt_idx) to build complete TrajectoryRecords.
    """
    # Step 1: Index all attempt metadata by (task_id, attempt_idx) for fast lookup.
    # Example entry: ("LCB/3402", 0) -> {"passed": false, "failure_type": "runtime", ...}
    meta: dict[tuple[str, int], dict] = {}
    jsonl_path = run_dir / "attempts.jsonl"
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            meta[(rec["task_id"], rec["attempt_idx"])] = rec

    # Step 2: Load hidden states from .npz files and join with metadata
    hs_dir = run_dir / "hidden_states"
    records: list[TrajectoryRecord] = []

    for npz_path in sorted(hs_dir.glob("*.npz")):
        # Convert filename back to task_id (e.g., "LCB_3402.npz" -> "LCB/3402")
        task_id = _task_id_from_filename(npz_path.name)
        data = np.load(npz_path)

        # Extract hidden-state vectors for the requested layer only.
        # A single .npz may contain many keys (all layers, all attempts);
        # we filter to just attempt_*_layer_{LAYER}.
        attempt_vectors: dict[int, np.ndarray] = {}
        for key in data.files:
            m = _ATTEMPT_LAYER_RE.match(key)
            if m and int(m.group(2)) == layer:
                attempt_idx = int(m.group(1))
                attempt_vectors[attempt_idx] = data[key]  # shape: (2560,)

        if not attempt_vectors:
            continue

        # Join each attempt's hidden state with its metadata from attempts.jsonl.
        # If metadata is missing for an attempt (shouldn't happen), skip it.
        attempts: list[AttemptInfo] = []
        for idx in sorted(attempt_vectors):
            m_rec = meta.get((task_id, idx))
            if m_rec is None:
                continue
            attempts.append(AttemptInfo(
                attempt_idx=idx,
                hidden_state=attempt_vectors[idx],
                passed=m_rec["passed"],
                failure_type=m_rec["failure_type"],
                completion_tokens=m_rec["completion_tokens"],
                prompt_tokens=m_rec["prompt_tokens"],
            ))

        if not attempts:
            continue

        # Detect repetition loops: the model generates the same broken output
        # every time. Symptoms: same failure_type across ALL attempts AND every
        # attempt hits the token limit (768). These tasks have near-zero
        # movement in representation space — the model isn't learning from errors.
        failure_types = set(a.failure_type for a in attempts)
        all_maxed = all(a.completion_tokens == TOKEN_LIMIT for a in attempts)
        is_loop = (len(failure_types) == 1 and all_maxed
                    and not any(a.passed for a in attempts))

        records.append(TrajectoryRecord(
            task_id=task_id,
            attempts=attempts,
            eventually_passes=any(a.passed for a in attempts),
            is_repetition_loop=is_loop,
        ))

    return records


# ── Printing helpers ─────────────────────────────────────────────────────────

def print_data_summary(records: list[TrajectoryRecord]) -> None:
    """Print overview of loaded data: counts, pass rates, trajectory lengths."""
    n = len(records)
    n_pass = sum(1 for r in records if r.eventually_passes)
    n_loops = sum(1 for r in records if r.is_repetition_loop)
    total_attempts = sum(len(r.attempts) for r in records)

    # Count how many tasks have each trajectory length (1, 2, 3, 4, 5)
    length_counts: dict[int, int] = defaultdict(int)
    for r in records:
        length_counts[len(r.attempts)] += 1

    print()
    print("== Data summary ===============================================")
    print(f"  Tasks            : {n}")
    print(f"  Eventually pass  : {n_pass} ({n_pass/n:.1%})")
    print(f"  Never pass       : {n - n_pass} ({(n-n_pass)/n:.1%})")
    print(f"  Repetition loops : {n_loops} ({n_loops/n:.1%}) — tagged, not removed")
    print(f"  Total attempts   : {total_attempts}")
    print(f"  Layer            : {LAYER}")
    print(f"  Hidden dim       : {records[0].attempts[0].hidden_state.shape[0]}")
    print(f"  Trajectory length distribution:")
    for length in sorted(length_counts):
        cnt = length_counts[length]
        print(f"    Length {length}: {cnt} tasks ({cnt/n:.0%})")
    print("================================================================")
    print()


def _section(title: str) -> None:
    """Print a visual section header to make console output scannable."""
    print()
    print(f"== {title} " + "=" * max(0, 60 - len(title) - 4))


# ── Analysis 1: Per-Attempt Classifier ───────────────────────────────────────

def analysis_1_classifier(
    records: list[TrajectoryRecord],
    seed: int,
) -> dict:
    """Train pass/fail classifiers on hidden states.

    Two modes:
      A) Pooled: mix ALL attempts (0-4) into one dataset, train one classifier.
         This tests: "regardless of attempt index, can the hidden state predict
         whether the model is about to generate correct code?"
      B) Per-attempt: train a separate classifier for each attempt index.
         This tests: "does the model's self-knowledge improve after seeing errors?"
         If AUC increases at attempt 1+ vs attempt 0, error feedback is helping
         the model's internal state become more informative.

    Uses the existing train_probe() pipeline: StandardScaler -> PCA -> LogisticRegression.
    Repetition-loop tasks are INCLUDED here (they're valid pass/fail samples).
    """
    _section("Analysis 1: Per-Attempt Classifier")

    # Flatten all attempts into parallel arrays for sklearn
    all_X: list[np.ndarray] = []   # hidden-state vectors
    all_y: list[int] = []          # 1=pass, 0=fail
    all_idx: list[int] = []        # attempt index (0, 1, 2, 3, 4)

    for rec in records:
        for att in rec.attempts:
            all_X.append(att.hidden_state)
            all_y.append(int(att.passed))
            all_idx.append(att.attempt_idx)

    X = np.stack(all_X).astype(np.float32)  # shape: (n_samples, 2560)
    y = np.array(all_y, dtype=np.int32)     # shape: (n_samples,)
    idx = np.array(all_idx)

    n_pass = int(y.sum())
    n_fail = len(y) - n_pass
    print(f"  Pooled dataset: {len(y)} samples "
          f"(pass={n_pass}, fail={n_fail}, pass_rate={n_pass/len(y):.1%})")
    print(f"  NOTE: dataset is {'imbalanced' if n_pass/len(y) < 0.3 else 'balanced'} "
          f"— interpret accuracy with caution if imbalanced")

    metrics: dict = {}

    # --- A) Pooled classifier (all attempts mixed) ---
    # Stratified split ensures train/test have same pass/fail ratio.
    # Need at least 2 samples per class for stratification to work.
    if n_pass >= 2 and n_fail >= 2:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed,
        )
        result, _ = train_probe(X_tr, y_tr, X_te, y_te, seed=seed,
                                layer_label="pooled")
        metrics["pooled"] = {
            "auc": result.test_auc, "accuracy": result.test_accuracy,
            "f1": result.test_f1, "cv_auc": result.cv_auc_mean,
            "n_train": result.n_train, "n_test": result.n_test,
            "n_pca": result.n_features_pca,
            "pass_rate_train": float(y_tr.sum() / len(y_tr)),
            "pass_rate_test": float(y_te.sum() / len(y_te)),
        }
        print(f"  Pooled  => CV AUC={result.cv_auc_mean:.3f}  "
              f"Test AUC={result.test_auc:.3f}  Acc={result.test_accuracy:.3f}  "
              f"F1={result.test_f1:.3f}  PCA_k={result.n_features_pca}")
        print(f"           Train pass_rate={y_tr.sum()/len(y_tr):.1%}  "
              f"Test pass_rate={y_te.sum()/len(y_te):.1%}")

    # --- B) Per-attempt-index classifiers ---
    # Train separate classifiers for attempt 0, attempt 1, etc.
    # This reveals if the model's internal signal changes with more context.
    print()
    print(f"  {'Attempt':<8} {'N':>4} {'Pass%':>6} {'CV AUC':>8} {'Test AUC':>9} "
          f"{'Acc':>6} {'F1':>6}")
    print("  " + "-" * 55)

    for attempt_i in sorted(set(idx)):
        mask = idx == attempt_i
        X_i, y_i = X[mask], y[mask]
        n_p = int(y_i.sum())
        n_f = len(y_i) - n_p

        # Skip if not enough samples for stratified split + CV
        if n_p < 2 or n_f < 2 or len(y_i) < 10:
            print(f"  {attempt_i:<8} {len(y_i):>4} {n_p/len(y_i):>5.1%}  "
                  f"SKIPPED (too few samples or classes)")
            continue

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_i, y_i, test_size=0.2, stratify=y_i, random_state=seed,
        )
        result, _ = train_probe(X_tr, y_tr, X_te, y_te, seed=seed,
                                layer_label=f"att_{attempt_i}")
        metrics[f"attempt_{attempt_i}"] = {
            "auc": result.test_auc, "accuracy": result.test_accuracy,
            "f1": result.test_f1, "cv_auc": result.cv_auc_mean,
            "n": len(y_i), "pass_rate": float(n_p / len(y_i)),
        }
        print(f"  {attempt_i:<8} {len(y_i):>4} {n_p/len(y_i):>5.1%}  "
              f"{result.cv_auc_mean:>7.3f}  {result.test_auc:>8.3f}  "
              f"{result.test_accuracy:>5.3f}  {result.test_f1:>5.3f}")

    return metrics


# ── Analysis 2: Trajectory Direction (RepE-style) ───────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0.0 if either is zero-norm."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def analysis_2_direction(records: list[TrajectoryRecord]) -> dict:
    """Find the "repair-success direction" in representation space.

    CORE IDEA (from RepE / Representation Engineering):
    When the model reads error feedback and is about to generate a fix, its
    hidden state shifts from h_k to h_{k+1}. We compute this shift (delta):
        delta = h_{k+1} - h_k

    Some deltas lead to a successful fix, others to another failure. If there's
    a geometric structure, the successful deltas should point in a consistent
    direction that differs from the failed deltas.

    The "repair-success direction" is:
        mean(success_deltas) - mean(failure_deltas)

    This cancels out the shared movement (common processing of error text) and
    isolates the component specific to "understanding vs not understanding the fix."

    We compute this in two ways:
      Option B: per-transition (0->1, 1->2, etc.) — cleaner, no mixing of contexts
      Option A: pooled across all transitions — more data, stronger signal

    Repetition-loop tasks are EXCLUDED here — their near-zero deltas would
    dilute the direction signal.
    """
    _section("Analysis 2: Trajectory Direction (RepE-style)")

    # Collect deltas, grouped by transition index (0->1, 1->2, ...) and outcome.
    # We only compute deltas where the CURRENT attempt FAILED — if it passed,
    # there's no repair to analyze (the task is already done).
    per_transition: dict[int, list[tuple[np.ndarray, bool]]] = defaultdict(list)

    for rec in records:
        # Skip repetition loops — they have near-identical hidden states
        # across attempts, so their deltas are near-zero noise
        if rec.is_repetition_loop:
            continue

        for i in range(len(rec.attempts) - 1):
            curr = rec.attempts[i]
            nxt = rec.attempts[i + 1]

            # Only compute delta for failed attempts that lead to a repair
            if curr.passed:
                continue
            if curr.completion_tokens == TOKEN_LIMIT:
                continue

            # delta = how the hidden state changed after receiving error feedback
            delta = nxt.hidden_state - curr.hidden_state
            # Label: did the NEXT attempt (after the shift) pass or fail?
            per_transition[curr.attempt_idx].append((delta, nxt.passed))

    # ---- Option B: separate direction per transition step ----
    # This is the cleaner analysis because transitions at different steps have
    # different prompt structures (attempt 0->1 has less context than 2->3).
    print("  Option B -- Per-transition analysis:")
    print(f"  {'Trans':<8} {'N_succ':>6} {'N_fail':>6} {'Dir norm':>9}")
    print("  " + "-" * 35)

    transition_directions: dict[int, np.ndarray] = {}
    metrics: dict = {"per_transition": {}, "pooled": {}}

    for trans_idx in sorted(per_transition):
        items = per_transition[trans_idx]
        # Split deltas by whether the next attempt passed or failed
        succ_deltas = [d for d, p in items if p]      # deltas before a successful fix
        fail_deltas = [d for d, p in items if not p]   # deltas before another failure

        # Need both classes to compute a contrastive direction
        if not succ_deltas or not fail_deltas:
            print(f"  {trans_idx}->{trans_idx+1}  {len(succ_deltas):>6} "
                  f"{len(fail_deltas):>6}  SKIPPED (need both classes)")
            continue

        # The direction that separates successful from failed repair shifts
        mean_succ = np.mean(succ_deltas, axis=0)  # average "understanding" shift
        mean_fail = np.mean(fail_deltas, axis=0)   # average "not understanding" shift
        direction = mean_succ - mean_fail           # what's different about success
        norm = float(np.linalg.norm(direction))
        transition_directions[trans_idx] = direction

        metrics["per_transition"][f"{trans_idx}->{trans_idx+1}"] = {
            "n_success": len(succ_deltas),
            "n_failure": len(fail_deltas),
            "direction_norm": norm,
        }
        print(f"  {trans_idx}->{trans_idx+1}  {len(succ_deltas):>6} "
              f"{len(fail_deltas):>6}  {norm:>9.2f}")

    # ---- Consistency check ----
    # If the repair-success direction is a real geometric property (not noise),
    # it should be similar across transition steps. We measure this with cosine
    # similarity. cos > 0.7 = strong consistency, cos < 0.3 = different directions.
    dir_keys = sorted(transition_directions)
    if len(dir_keys) >= 2:
        print()
        print("  Consistency (cosine sim between transition directions):")
        for i in range(len(dir_keys)):
            for j in range(i + 1, len(dir_keys)):
                cos = _cosine_sim(
                    transition_directions[dir_keys[i]],
                    transition_directions[dir_keys[j]],
                )
                label = f"  {dir_keys[i]}->{dir_keys[i]+1} vs {dir_keys[j]}->{dir_keys[j]+1}"
                print(f"  {label}: cos={cos:.3f}")
                metrics["per_transition"][f"cos_{dir_keys[i]}v{dir_keys[j]}"] = cos

    # ---- Option A: pooled direction (all transitions mixed) ----
    # More data = more statistical power, but mixes different context lengths.
    # If the consistency check above shows cos > 0.7, pooling is justified.
    print()
    print("  Option A -- Pooled direction:")
    all_succ = [d for items in per_transition.values() for d, p in items if p]
    all_fail = [d for items in per_transition.values() for d, p in items if not p]

    repair_direction = np.zeros_like(records[0].attempts[0].hidden_state)
    if all_succ and all_fail:
        mean_succ = np.mean(all_succ, axis=0)
        mean_fail = np.mean(all_fail, axis=0)
        repair_direction = mean_succ - mean_fail
        norm = float(np.linalg.norm(repair_direction))
        print(f"  N_success={len(all_succ)}  N_failure={len(all_fail)}  "
              f"Direction norm={norm:.2f}")
        metrics["pooled"] = {
            "n_success": len(all_succ),
            "n_failure": len(all_fail),
            "direction_norm": norm,
        }
    else:
        print("  SKIPPED (need both success and failure deltas)")

    # Pass the direction vector to Analysis 3 via a private key (not serialized)
    metrics["_repair_direction"] = repair_direction
    return metrics


# ── Analysis 3: Convergence ─────────────────────────────────────────────────

def analysis_3_convergence(
    records: list[TrajectoryRecord],
    repair_direction: np.ndarray,
) -> dict:
    """Test whether successful trajectories converge while failed ones wander.

    Two sub-analyses:
      A) Directional consistency: for each task, compute cosine similarity between
         consecutive deltas (delta_k, delta_{k+1}). High cosine = the model keeps
         moving in the same direction. Low cosine = the model is changing direction
         (wandering). We compare consistency for tasks that eventually pass vs fail.

      B) Projection onto repair direction: at each step, project the delta onto
         the repair-success direction from Analysis 2. If successful tasks show
         increasing projection over steps, it means they're progressively aligning
         with the "understanding" direction — evidence of convergence toward a fix.

    Only uses tasks with 3+ attempts (need at least 2 deltas to measure consistency).
    Repetition-loop tasks are EXCLUDED (they have near-zero deltas, meaningless
    cosine similarity).
    """
    _section("Analysis 3: Convergence")

    # Filter to tasks with 3+ attempts and exclude repetition loops
    long_records = [r for r in records
                    if len(r.attempts) >= 3 and not r.is_repetition_loop]
    print(f"  Tasks with 3+ attempts (excl. loops): {len(long_records)}")

    if not long_records:
        print("  SKIPPED (no qualifying tasks)")
        return {}

    # Split by final outcome for comparison
    pass_recs = [r for r in long_records if r.eventually_passes]
    fail_recs = [r for r in long_records if not r.eventually_passes]
    print(f"  Eventually pass: {len(pass_recs)}  Never pass: {len(fail_recs)}")

    metrics: dict = {}

    # ---- A) Directional consistency ----
    # For each task, compute consecutive deltas and measure if they point the
    # same way (high cosine) or diverge (low cosine).
    for label, group in [("pass", pass_recs), ("fail", fail_recs)]:
        cos_sims: list[float] = []
        for rec in group:
            # Compute all consecutive deltas for this task
            deltas = [
                rec.attempts[i + 1].hidden_state - rec.attempts[i].hidden_state
                for i in range(len(rec.attempts) - 1)
            ]
            # Compare each pair of consecutive deltas
            for j in range(len(deltas) - 1):
                cos_sims.append(_cosine_sim(deltas[j], deltas[j + 1]))

        if cos_sims:
            mean_cos = float(np.mean(cos_sims))
            std_cos = float(np.std(cos_sims))
            metrics[f"consistency_{label}"] = {
                "mean": mean_cos, "std": std_cos, "n": len(cos_sims),
            }
            print(f"  Directional consistency ({label}): "
                  f"cos={mean_cos:.3f} +/- {std_cos:.3f} (n={len(cos_sims)})")

    # ---- B) Projection onto repair direction at each step ----
    # The repair direction (from Analysis 2) represents "the way a hidden state
    # moves when the model is about to produce correct code." By projecting each
    # step's delta onto this direction, we measure how much each step aligns
    # with the "success shift." Higher projection = more aligned with success.
    dir_norm = np.linalg.norm(repair_direction)
    if dir_norm > 0:
        unit_dir = repair_direction / dir_norm  # normalize to unit vector
        print()
        print("  Projection of deltas onto repair direction (by step):")
        print(f"  {'Step':<8} {'Pass proj':>10} {'Fail proj':>10}")
        print("  " + "-" * 30)

        max_steps = max(len(r.attempts) for r in long_records) - 1
        for step in range(max_steps):
            pass_projs: list[float] = []
            fail_projs: list[float] = []

            for rec in long_records:
                if step + 1 >= len(rec.attempts):
                    continue  # this task doesn't have enough attempts
                delta = (rec.attempts[step + 1].hidden_state
                         - rec.attempts[step].hidden_state)
                # Scalar projection: how much does this delta align with success?
                proj = float(np.dot(delta, unit_dir))

                if rec.eventually_passes:
                    pass_projs.append(proj)
                else:
                    fail_projs.append(proj)

            pass_mean = f"{np.mean(pass_projs):.2f}" if pass_projs else "n/a"
            fail_mean = f"{np.mean(fail_projs):.2f}" if fail_projs else "n/a"
            print(f"  {step}->{step+1}    {pass_mean:>10} {fail_mean:>10}")
            metrics[f"proj_step_{step}"] = {
                "pass_mean": float(np.mean(pass_projs)) if pass_projs else None,
                "fail_mean": float(np.mean(fail_projs)) if fail_projs else None,
                "pass_n": len(pass_projs),
                "fail_n": len(fail_projs),
            }

    return metrics


# ── Analysis 4: PCA Trajectory Visualization ────────────────────────────────

def analysis_4_pca(
    records: list[TrajectoryRecord],
    output_dir: Path,
) -> dict:
    """Project all hidden states to 2D via PCA and plot repair trajectories.

    Each task becomes a connected path in PC1-PC2 space, showing how the
    model's internal state moves across repair attempts. We use three colors:
      - Green: tasks that eventually pass (successful repair trajectories)
      - Red: tasks that never pass (failed trajectories with real effort)
      - Gray: repetition-loop tasks (stuck, no real movement)

    Individual points are marked as circles (pass) or X (fail).
    """
    _section("Analysis 4: PCA Trajectory Visualization")

    # Stack all hidden states into one matrix for PCA.
    # We keep track of the order so we can map PCA results back to trajectories.
    all_vectors: list[np.ndarray] = []
    for rec in records:
        for att in rec.attempts:
            all_vectors.append(att.hidden_state)

    X = np.stack(all_vectors).astype(np.float32)  # shape: (total_attempts, 2560)

    # PCA reduces 2560 dims to 2 for visualization.
    # We lose most variance but can see global structure.
    pca = PCA(n_components=2, random_state=SEED)
    X_2d = pca.fit_transform(X)

    var_explained = pca.explained_variance_ratio_
    print(f"  PCA variance explained: PC1={var_explained[0]:.1%}, "
          f"PC2={var_explained[1]:.1%}")

    # ---- Plot trajectories ----
    fig, ax = plt.subplots(figsize=(10, 8))
    offset = 0  # tracks position in the flattened X_2d array

    for rec in records:
        n = len(rec.attempts)
        points = X_2d[offset:offset + n]  # this task's PCA-projected points
        offset += n

        # Color by task category
        if rec.is_repetition_loop:
            color, alpha = "#95a5a6", 0.3       # gray, faint
        elif rec.eventually_passes:
            color, alpha = "#2ecc71", 0.7        # green
        else:
            color, alpha = "#e74c3c", 0.4        # red

        # Draw the trajectory as a connected line
        ax.plot(points[:, 0], points[:, 1], color=color, alpha=alpha,
                linewidth=1.0, zorder=1)

        # Mark each attempt as a point
        for i, att in enumerate(rec.attempts):
            marker = "o" if att.passed else "x"  # circle = pass, X = fail
            size = 40 if att.passed else 25
            ax.scatter(points[i, 0], points[i, 1], c=color, marker=marker,
                       s=size, alpha=alpha, zorder=2, edgecolors="white",
                       linewidths=0.5)

            # Small label at the start of each trajectory (task number)
            if i == 0:
                ax.annotate(str(rec.task_id.split("/")[-1]),
                            (points[i, 0], points[i, 1]),
                            fontsize=5, alpha=0.4)

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)")
    ax.set_title(f"Repair Trajectories in PCA Space (layer {LAYER})")

    # Custom legend with all three categories
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2ecc71", label="Eventually passes"),
        Line2D([0], [0], color="#e74c3c", label="Never passes"),
        Line2D([0], [0], color="#95a5a6", label="Repetition loop"),
        Line2D([0], [0], marker="o", color="gray", label="Passed", markersize=6,
               linestyle="None"),
        Line2D([0], [0], marker="x", color="gray", label="Failed", markersize=6,
               linestyle="None"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_dir / "trajectory_pca.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'trajectory_pca.png'}")

    return {
        "pc1_variance": float(var_explained[0]),
        "pc2_variance": float(var_explained[1]),
    }


# ── Analysis 5: Distance to Success Region ──────────────────────────────────

def analysis_5_distance(
    records: list[TrajectoryRecord],
    output_dir: Path,
) -> dict:
    """Track how far each task's hidden state is from the "success centroid."

    The "success centroid" is the mean hidden state of tasks that pass on
    attempt 0 — these are tasks the model "knew" how to solve from the start.
    By measuring Euclidean distance to this centroid at each repair step, we
    test whether successful repair trajectories physically move toward the
    region of representation space associated with correct solutions.

    If the green line (eventually passes) trends downward = the model's
    representation is converging toward the success region as it repairs.
    If the red line (never passes) stays flat or diverges = no convergence.
    """
    _section("Analysis 5: Distance to Success Region")

    # Compute centroid from tasks that pass on the very first attempt.
    # These tasks represent "the model already knew the answer" — their
    # attempt-0 hidden states define the "success region."
    pass_vectors = [
        rec.attempts[0].hidden_state
        for rec in records
        if rec.attempts[0].passed
    ]

    if not pass_vectors:
        print("  SKIPPED (no tasks pass on attempt 0 — can't define success region)")
        return {}

    centroid = np.mean(pass_vectors, axis=0)  # shape: (2560,)
    print(f"  Success centroid computed from {len(pass_vectors)} attempt-0 passes")

    # Compute distance-to-centroid at each step for every task
    max_attempts = max(len(r.attempts) for r in records)

    # Group distance curves by outcome (and separate repetition loops)
    pass_curves: list[list[float]] = []
    fail_curves: list[list[float]] = []
    loop_curves: list[list[float]] = []

    for rec in records:
        distances = [
            float(np.linalg.norm(att.hidden_state - centroid))
            for att in rec.attempts
        ]
        if rec.is_repetition_loop:
            loop_curves.append(distances)
        elif rec.eventually_passes:
            pass_curves.append(distances)
        else:
            fail_curves.append(distances)

    # ---- Plot mean +/- std distance curves ----
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, curves, color in [
        ("Eventually passes", pass_curves, "#2ecc71"),
        ("Never passes", fail_curves, "#e74c3c"),
        ("Repetition loops", loop_curves, "#95a5a6"),
    ]:
        if not curves:
            continue
        # Pad shorter curves with NaN so numpy can compute column-wise mean/std
        padded = np.full((len(curves), max_attempts), np.nan)
        for i, c in enumerate(curves):
            padded[i, :len(c)] = c

        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        steps = np.arange(max_attempts)

        ax.plot(steps, mean, color=color, label=label, linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("Attempt index")
    ax.set_ylabel("Euclidean distance to success centroid")
    ax.set_title(f"Distance to Success Region (layer {LAYER})")
    ax.legend()
    ax.set_xticks(range(max_attempts))
    plt.tight_layout()

    fig.savefig(output_dir / "distance_to_success.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'distance_to_success.png'}")

    # ---- Print distance table ----
    metrics: dict = {}
    print()
    print(f"  {'Step':<6} {'Pass dist':>10} {'Fail dist':>10} {'Loop dist':>10}")
    print("  " + "-" * 38)
    for step in range(max_attempts):
        p_dists = [c[step] for c in pass_curves if step < len(c)]
        f_dists = [c[step] for c in fail_curves if step < len(c)]
        l_dists = [c[step] for c in loop_curves if step < len(c)]
        p_mean = float(np.mean(p_dists)) if p_dists else float("nan")
        f_mean = float(np.mean(f_dists)) if f_dists else float("nan")
        l_mean = float(np.mean(l_dists)) if l_dists else float("nan")
        print(f"  {step:<6} {p_mean:>10.1f} {f_mean:>10.1f} {l_mean:>10.1f}")
        metrics[f"step_{step}"] = {
            "pass_mean_dist": p_mean if p_dists else None,
            "fail_mean_dist": f_mean if f_dists else None,
            "loop_mean_dist": l_mean if l_dists else None,
            "pass_n": len(p_dists), "fail_n": len(f_dists), "loop_n": len(l_dists),
        }

    return metrics


# ── Analysis 6: Permutation Test on 0->1 Repair Direction ──────────────────

def _collect_transition_deltas(
    records: list[TrajectoryRecord],
    trans_idx: int = 0,
) -> tuple[list[np.ndarray], list[bool]]:
    """Collect hidden-state deltas for a specific transition step.

    Returns (deltas, labels) where labels[i] = True if the NEXT attempt passed.
    Only includes transitions where the current attempt FAILED (no repair to
    analyze if it already passed). Excludes repetition-loop tasks and tasks
    where the current attempt hit the token limit (768), which flood the
    repair prompt with truncated code and distort delta_prompt_tokens.
    """
    deltas: list[np.ndarray] = []
    labels: list[bool] = []  # True = next attempt passed
    for rec in records:
        if rec.is_repetition_loop:
            continue
        if len(rec.attempts) <= trans_idx + 1:
            continue
        curr = rec.attempts[trans_idx]
        nxt = rec.attempts[trans_idx + 1]
        if curr.passed:
            continue
        if curr.completion_tokens == TOKEN_LIMIT:
            continue
        deltas.append(nxt.hidden_state - curr.hidden_state)
        labels.append(nxt.passed)
    return deltas, labels


def analysis_6_permutation_test(
    records: list[TrajectoryRecord],
    seed: int,
    output_dir: Path,
) -> dict:
    """Permutation test: is the 0->1 repair direction norm significant?

    Shuffles pass/fail labels 1000 times, recomputes ||mean_success - mean_failure||
    each time, and checks where the real norm falls in the null distribution.
    """
    _section("Analysis 6: Permutation Test (0->1 direction)")

    deltas, labels = _collect_transition_deltas(records, trans_idx=0)
    if not deltas:
        print("  SKIPPED (no 0->1 transitions)")
        return {}

    deltas_arr = np.stack(deltas)  # shape: (N, hidden_dim)
    labels_arr = np.array(labels)
    n_succ = int(labels_arr.sum())
    n_fail = len(labels_arr) - n_succ

    if n_succ == 0 or n_fail == 0:
        print("  SKIPPED (need both pass and fail labels)")
        return {}

    # Real direction norm
    mean_succ = deltas_arr[labels_arr].mean(axis=0)
    mean_fail = deltas_arr[~labels_arr].mean(axis=0)
    real_norm = float(np.linalg.norm(mean_succ - mean_fail))
    print(f"  N_success={n_succ}  N_failure={n_fail}")
    print(f"  Real direction norm: {real_norm:.2f}")

    # Permutation null distribution
    rng = np.random.default_rng(seed)
    n_perms = 1000
    perm_norms = np.empty(n_perms)
    for i in range(n_perms):
        shuffled = rng.permutation(labels_arr)
        ms = deltas_arr[shuffled].mean(axis=0)
        mf = deltas_arr[~shuffled].mean(axis=0)
        perm_norms[i] = np.linalg.norm(ms - mf)

    p_value = float((perm_norms >= real_norm).mean())
    print(f"  Permutation norms: mean={perm_norms.mean():.2f}  "
          f"std={perm_norms.std():.2f}  max={perm_norms.max():.2f}")
    print(f"  p-value: {p_value:.4f}  "
          f"({'significant' if p_value < 0.05 else 'NOT significant'} at alpha=0.05)")

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm_norms, bins=50, color="#95a5a6", alpha=0.7, label="Permuted norms")
    ax.axvline(real_norm, color="#e74c3c", linewidth=2,
               label=f"Real norm = {real_norm:.2f}")
    ax.set_xlabel("||mean_success - mean_failure||")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation Test: 0->1 Repair Direction (n={len(deltas)}, "
                 f"p={p_value:.4f})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "permutation_test_0to1.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'permutation_test_0to1.png'}")

    return {
        "real_norm": real_norm,
        "perm_mean": float(perm_norms.mean()),
        "perm_std": float(perm_norms.std()),
        "perm_max": float(perm_norms.max()),
        "p_value": p_value,
        "n_permutations": n_perms,
        "n_success": n_succ,
        "n_failure": n_fail,
    }


# ── Analysis 7: PCA Reduction Helper ──────────────────────────────────────

def _build_pca_reduced_records(
    records: list[TrajectoryRecord],
    n_components: int = 100,
) -> tuple[list[TrajectoryRecord], PCA]:
    """Fit PCA on attempt-0 states only, transform all attempts.

    This factors out within-attempt variance by defining the coordinate system
    from the initial prompt states. Attempt-1+ states are projected INTO this
    space, so any movement we see is real geometric shift, not an artifact of
    PCA fitting on mixed-context data.

    Returns (records_pca, pca_object).
    """
    _section("PCA Reduction: fit on attempt-0, transform all")

    X_att0 = np.stack([r.attempts[0].hidden_state for r in records]).astype(np.float32)

    pca = PCA(n_components=n_components, random_state=SEED)
    pca.fit(X_att0)

    var_total = float(pca.explained_variance_ratio_.sum())
    print(f"  Fitted on {len(X_att0)} attempt-0 states")
    print(f"  {n_components} components capture {var_total:.1%} of attempt-0 variance")

    new_records: list[TrajectoryRecord] = []
    for rec in records:
        new_attempts: list[AttemptInfo] = []
        for att in rec.attempts:
            reduced = pca.transform(att.hidden_state.reshape(1, -1).astype(np.float32))[0]
            new_attempts.append(AttemptInfo(
                attempt_idx=att.attempt_idx,
                hidden_state=reduced,
                passed=att.passed,
                failure_type=att.failure_type,
                completion_tokens=att.completion_tokens,
                prompt_tokens=att.prompt_tokens,
            ))
        new_records.append(TrajectoryRecord(
            task_id=rec.task_id,
            attempts=new_attempts,
            eventually_passes=rec.eventually_passes,
            is_repetition_loop=rec.is_repetition_loop,
        ))

    return new_records, pca


# ── Analysis 8: Direction in PCA-Reduced Space ────────────────────────────

def analysis_8_direction_pca(records_pca: list[TrajectoryRecord]) -> dict:
    """Redo Analysis 2 (RepE direction) in PCA-reduced space.

    If the repair-success direction survives PCA reduction (fitted on attempt-0
    states), it's encoding repair quality, not prompt structure.
    """
    _section("Analysis 8: Direction in PCA-Reduced Space")

    per_transition: dict[int, list[tuple[np.ndarray, bool]]] = defaultdict(list)
    for rec in records_pca:
        if rec.is_repetition_loop:
            continue
        for i in range(len(rec.attempts) - 1):
            curr = rec.attempts[i]
            nxt = rec.attempts[i + 1]
            if curr.passed:
                continue
            if curr.completion_tokens == TOKEN_LIMIT:
                continue
            delta = nxt.hidden_state - curr.hidden_state
            per_transition[curr.attempt_idx].append((delta, nxt.passed))

    print("  Per-transition (PCA-reduced):")
    print(f"  {'Trans':<8} {'N_succ':>6} {'N_fail':>6} {'Dir norm':>9}")
    print("  " + "-" * 35)

    transition_directions: dict[int, np.ndarray] = {}
    metrics: dict = {"per_transition": {}, "pooled": {}}

    for trans_idx in sorted(per_transition):
        items = per_transition[trans_idx]
        succ_deltas = [d for d, p in items if p]
        fail_deltas = [d for d, p in items if not p]

        if not succ_deltas or not fail_deltas:
            print(f"  {trans_idx}->{trans_idx+1}  {len(succ_deltas):>6} "
                  f"{len(fail_deltas):>6}  SKIPPED")
            continue

        mean_succ = np.mean(succ_deltas, axis=0)
        mean_fail = np.mean(fail_deltas, axis=0)
        direction = mean_succ - mean_fail
        norm = float(np.linalg.norm(direction))
        transition_directions[trans_idx] = direction

        metrics["per_transition"][f"{trans_idx}->{trans_idx+1}"] = {
            "n_success": len(succ_deltas),
            "n_failure": len(fail_deltas),
            "direction_norm": norm,
        }
        print(f"  {trans_idx}->{trans_idx+1}  {len(succ_deltas):>6} "
              f"{len(fail_deltas):>6}  {norm:>9.2f}")

    # Consistency check
    dir_keys = sorted(transition_directions)
    if len(dir_keys) >= 2:
        print()
        print("  Consistency (cosine sim between PCA-reduced directions):")
        for i in range(len(dir_keys)):
            for j in range(i + 1, len(dir_keys)):
                cos = _cosine_sim(
                    transition_directions[dir_keys[i]],
                    transition_directions[dir_keys[j]],
                )
                label = f"  {dir_keys[i]}->{dir_keys[i]+1} vs {dir_keys[j]}->{dir_keys[j]+1}"
                print(f"  {label}: cos={cos:.3f}")
                metrics["per_transition"][f"cos_{dir_keys[i]}v{dir_keys[j]}"] = cos

    # Pooled direction
    print()
    print("  Pooled direction (PCA-reduced):")
    all_succ = [d for items in per_transition.values() for d, p in items if p]
    all_fail = [d for items in per_transition.values() for d, p in items if not p]

    dim = records_pca[0].attempts[0].hidden_state.shape[0]
    repair_direction = np.zeros(dim)
    if all_succ and all_fail:
        mean_succ = np.mean(all_succ, axis=0)
        mean_fail = np.mean(all_fail, axis=0)
        repair_direction = mean_succ - mean_fail
        norm = float(np.linalg.norm(repair_direction))
        print(f"  N_success={len(all_succ)}  N_failure={len(all_fail)}  "
              f"Direction norm={norm:.2f}")
        metrics["pooled"] = {
            "n_success": len(all_succ),
            "n_failure": len(all_fail),
            "direction_norm": norm,
        }
    else:
        print("  SKIPPED (need both success and failure deltas)")

    metrics["_repair_direction"] = repair_direction
    return metrics


# ── Analysis 9: Convergence in PCA-Reduced Space ─────────────────────────

def analysis_9_convergence_pca(
    records_pca: list[TrajectoryRecord],
    repair_direction_pca: np.ndarray,
) -> dict:
    """Redo Analysis 3 (convergence) in PCA-reduced space."""
    _section("Analysis 9: Convergence in PCA-Reduced Space")

    long_records = [r for r in records_pca
                    if len(r.attempts) >= 3 and not r.is_repetition_loop]
    print(f"  Tasks with 3+ attempts (excl. loops): {len(long_records)}")

    if not long_records:
        print("  SKIPPED (no qualifying tasks)")
        return {}

    pass_recs = [r for r in long_records if r.eventually_passes]
    fail_recs = [r for r in long_records if not r.eventually_passes]
    print(f"  Eventually pass: {len(pass_recs)}  Never pass: {len(fail_recs)}")

    metrics: dict = {}

    # A) Directional consistency
    for label, group in [("pass", pass_recs), ("fail", fail_recs)]:
        cos_sims: list[float] = []
        for rec in group:
            deltas = [
                rec.attempts[i + 1].hidden_state - rec.attempts[i].hidden_state
                for i in range(len(rec.attempts) - 1)
            ]
            for j in range(len(deltas) - 1):
                cos_sims.append(_cosine_sim(deltas[j], deltas[j + 1]))

        if cos_sims:
            mean_cos = float(np.mean(cos_sims))
            std_cos = float(np.std(cos_sims))
            metrics[f"consistency_{label}"] = {
                "mean": mean_cos, "std": std_cos, "n": len(cos_sims),
            }
            print(f"  Directional consistency ({label}): "
                  f"cos={mean_cos:.3f} +/- {std_cos:.3f} (n={len(cos_sims)})")

    # B) Projection onto PCA-reduced repair direction
    dir_norm = np.linalg.norm(repair_direction_pca)
    if dir_norm > 0:
        unit_dir = repair_direction_pca / dir_norm
        print()
        print("  Projection of deltas onto PCA repair direction (by step):")
        print(f"  {'Step':<8} {'Pass proj':>10} {'Fail proj':>10}")
        print("  " + "-" * 30)

        max_steps = max(len(r.attempts) for r in long_records) - 1
        for step in range(max_steps):
            pass_projs: list[float] = []
            fail_projs: list[float] = []
            for rec in long_records:
                if step + 1 >= len(rec.attempts):
                    continue
                delta = (rec.attempts[step + 1].hidden_state
                         - rec.attempts[step].hidden_state)
                proj = float(np.dot(delta, unit_dir))
                if rec.eventually_passes:
                    pass_projs.append(proj)
                else:
                    fail_projs.append(proj)

            pass_mean = f"{np.mean(pass_projs):.2f}" if pass_projs else "n/a"
            fail_mean = f"{np.mean(fail_projs):.2f}" if fail_projs else "n/a"
            print(f"  {step}->{step+1}    {pass_mean:>10} {fail_mean:>10}")
            metrics[f"proj_step_{step}"] = {
                "pass_mean": float(np.mean(pass_projs)) if pass_projs else None,
                "fail_mean": float(np.mean(fail_projs)) if fail_projs else None,
                "pass_n": len(pass_projs),
                "fail_n": len(fail_projs),
            }

    return metrics


# ── Analysis 10: Distance to Success (PCA, first-pass centroid) ───────────

def analysis_10_distance_pca(
    records_pca: list[TrajectoryRecord],
    output_dir: Path,
) -> dict:
    """Distance-to-success in PCA-reduced space with improved centroid.

    Centroid is computed from the hidden state at the FIRST passing attempt
    for each task (not just attempt-0 passes). This captures "the model state
    when it finally understood the fix."
    """
    _section("Analysis 10: Distance to Success (PCA, first-pass centroid)")

    # Collect hidden state at the FIRST passing attempt for each passing task
    first_pass_vectors: list[np.ndarray] = []
    for rec in records_pca:
        if rec.eventually_passes:
            for att in rec.attempts:
                if att.passed:
                    first_pass_vectors.append(att.hidden_state)
                    break

    if not first_pass_vectors:
        print("  SKIPPED (no passing tasks)")
        return {}

    centroid = np.mean(first_pass_vectors, axis=0)
    print(f"  First-pass centroid from {len(first_pass_vectors)} tasks")

    max_attempts = max(len(r.attempts) for r in records_pca)

    pass_curves: list[list[float]] = []
    fail_curves: list[list[float]] = []
    loop_curves: list[list[float]] = []

    for rec in records_pca:
        distances = [
            float(np.linalg.norm(att.hidden_state - centroid))
            for att in rec.attempts
        ]
        if rec.is_repetition_loop:
            loop_curves.append(distances)
        elif rec.eventually_passes:
            pass_curves.append(distances)
        else:
            fail_curves.append(distances)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, curves, color in [
        ("Eventually passes", pass_curves, "#2ecc71"),
        ("Never passes", fail_curves, "#e74c3c"),
        ("Repetition loops", loop_curves, "#95a5a6"),
    ]:
        if not curves:
            continue
        padded = np.full((len(curves), max_attempts), np.nan)
        for i, c in enumerate(curves):
            padded[i, :len(c)] = c
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        steps = np.arange(max_attempts)
        ax.plot(steps, mean, color=color, label=label, linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("Attempt index")
    ax.set_ylabel("Euclidean distance to first-pass centroid (PCA)")
    ax.set_title(f"Distance to Success Region - PCA-reduced (layer {LAYER})")
    ax.legend()
    ax.set_xticks(range(max_attempts))
    plt.tight_layout()
    fig.savefig(output_dir / "distance_to_success_pca.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'distance_to_success_pca.png'}")

    # Table
    metrics: dict = {}
    print()
    print(f"  {'Step':<6} {'Pass dist':>10} {'Fail dist':>10} {'Loop dist':>10}")
    print("  " + "-" * 38)
    for step in range(max_attempts):
        p_dists = [c[step] for c in pass_curves if step < len(c)]
        f_dists = [c[step] for c in fail_curves if step < len(c)]
        l_dists = [c[step] for c in loop_curves if step < len(c)]
        p_mean = float(np.mean(p_dists)) if p_dists else float("nan")
        f_mean = float(np.mean(f_dists)) if f_dists else float("nan")
        l_mean = float(np.mean(l_dists)) if l_dists else float("nan")
        print(f"  {step:<6} {p_mean:>10.1f} {f_mean:>10.1f} {l_mean:>10.1f}")
        metrics[f"step_{step}"] = {
            "pass_mean_dist": p_mean if p_dists else None,
            "fail_mean_dist": f_mean if f_dists else None,
            "loop_mean_dist": l_mean if l_dists else None,
            "pass_n": len(p_dists), "fail_n": len(f_dists), "loop_n": len(l_dists),
        }

    return metrics


# ── Analysis 11: Per-Task Projection Histogram ────────────────────────────

def analysis_11_projection_histogram(
    records: list[TrajectoryRecord],
    repair_direction: np.ndarray,
    output_dir: Path,
) -> dict:
    """Histogram of 0->1 delta projections onto the repair direction.

    Shows whether pass and fail deltas are well-separated or heavily overlapping
    when projected onto the contrastive direction from Analysis 2.
    """
    _section("Analysis 11: Projection Histogram (0->1)")

    dir_norm = np.linalg.norm(repair_direction)
    if dir_norm == 0:
        print("  SKIPPED (repair direction is zero)")
        return {}
    unit_dir = repair_direction / dir_norm

    deltas, labels = _collect_transition_deltas(records, trans_idx=0)
    if not deltas:
        print("  SKIPPED (no 0->1 transitions)")
        return {}

    pass_projs: list[float] = []
    fail_projs: list[float] = []
    for d, passed in zip(deltas, labels):
        proj = float(np.dot(d, unit_dir))
        if passed:
            pass_projs.append(proj)
        else:
            fail_projs.append(proj)

    if not pass_projs or not fail_projs:
        print("  SKIPPED (need both pass and fail projections)")
        return {}

    # Cohen's d
    pass_arr = np.array(pass_projs)
    fail_arr = np.array(fail_projs)
    pooled_std = np.sqrt(
        ((len(pass_arr) - 1) * pass_arr.std(ddof=1)**2
         + (len(fail_arr) - 1) * fail_arr.std(ddof=1)**2)
        / (len(pass_arr) + len(fail_arr) - 2)
    )
    cohens_d = float((pass_arr.mean() - fail_arr.mean()) / pooled_std) if pooled_std > 0 else 0.0

    print(f"  Pass projections: mean={pass_arr.mean():.2f}  std={pass_arr.std():.2f}  n={len(pass_arr)}")
    print(f"  Fail projections: mean={fail_arr.mean():.2f}  std={fail_arr.std():.2f}  n={len(fail_arr)}")
    print(f"  Cohen's d: {cohens_d:.3f}")

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(pass_arr.min(), fail_arr.min()),
        max(pass_arr.max(), fail_arr.max()),
        40,
    )
    ax.hist(fail_projs, bins=bins, color="#e74c3c", alpha=0.6, label=f"Fail (n={len(fail_arr)})")
    ax.hist(pass_projs, bins=bins, color="#2ecc71", alpha=0.6, label=f"Pass (n={len(pass_arr)})")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Projection onto repair direction")
    ax.set_ylabel("Count")
    ax.set_title(f"0->1 Delta Projections (Cohen's d = {cohens_d:.2f})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "projection_histogram_0to1.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'projection_histogram_0to1.png'}")

    return {
        "pass_mean": float(pass_arr.mean()),
        "pass_std": float(pass_arr.std()),
        "fail_mean": float(fail_arr.mean()),
        "fail_std": float(fail_arr.std()),
        "cohens_d": cohens_d,
        "n_pass": len(pass_arr),
        "n_fail": len(fail_arr),
    }


# ── Analysis 12: Focused PCA Plot (Attempt 0 vs 1) ───────────────────────

def analysis_12_focused_pca(
    records: list[TrajectoryRecord],
    output_dir: Path,
) -> dict:
    """PCA plot of just attempt-0 and attempt-1, colored by attempt-1 outcome.

    PCA is fitted on attempt-0 states only (same philosophy as Analysis 7).
    Arrows show the 0->1 transition direction for each task.
    """
    _section("Analysis 12: Focused PCA (attempt 0 vs 1)")

    # Collect tasks with 2+ attempts, excluding repetition loops
    tasks: list[tuple[np.ndarray, np.ndarray, bool]] = []  # (h0, h1, att1_passed)
    for rec in records:
        if rec.is_repetition_loop or len(rec.attempts) < 2:
            continue
        h0 = rec.attempts[0].hidden_state
        h1 = rec.attempts[1].hidden_state
        tasks.append((h0, h1, rec.attempts[1].passed))

    if len(tasks) < 5:
        print(f"  SKIPPED (only {len(tasks)} qualifying tasks)")
        return {}

    h0_all = np.stack([t[0] for t in tasks]).astype(np.float32)
    h1_all = np.stack([t[1] for t in tasks]).astype(np.float32)
    att1_passed = [t[2] for t in tasks]

    # PCA fitted on attempt-0 only
    pca = PCA(n_components=2, random_state=SEED)
    pca.fit(h0_all)

    h0_2d = pca.transform(h0_all)
    h1_2d = pca.transform(h1_all)

    var = pca.explained_variance_ratio_
    print(f"  PCA (fitted on attempt-0): PC1={var[0]:.1%}  PC2={var[1]:.1%}")
    print(f"  Tasks plotted: {len(tasks)} "
          f"(attempt-1 pass: {sum(att1_passed)}, fail: {sum(not p for p in att1_passed)})")

    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(len(tasks)):
        color = "#2ecc71" if att1_passed[i] else "#e74c3c"
        alpha = 0.7 if att1_passed[i] else 0.4

        # Attempt-0: circle
        ax.scatter(h0_2d[i, 0], h0_2d[i, 1], c=color, marker="o",
                   s=30, alpha=alpha, zorder=2, edgecolors="white", linewidths=0.5)
        # Attempt-1: triangle
        ax.scatter(h1_2d[i, 0], h1_2d[i, 1], c=color, marker="^",
                   s=30, alpha=alpha, zorder=2, edgecolors="white", linewidths=0.5)
        # Arrow
        ax.annotate("", xy=(h1_2d[i, 0], h1_2d[i, 1]),
                    xytext=(h0_2d[i, 0], h0_2d[i, 1]),
                    arrowprops=dict(arrowstyle="->", color=color, alpha=alpha * 0.6,
                                    linewidth=0.8))

    ax.set_xlabel(f"PC1 ({var[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} variance)")
    ax.set_title(f"Attempt 0 -> 1 Transitions (PCA on attempt-0, layer {LAYER})")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="gray", label="Attempt 0", markersize=6,
               linestyle="None"),
        Line2D([0], [0], marker="^", color="gray", label="Attempt 1", markersize=6,
               linestyle="None"),
        Line2D([0], [0], color="#2ecc71", label="Attempt 1 passes", linewidth=2),
        Line2D([0], [0], color="#e74c3c", label="Attempt 1 fails", linewidth=2),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    fig.savefig(output_dir / "focused_pca_0vs1.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'focused_pca_0vs1.png'}")

    return {
        "pc1_variance": float(var[0]),
        "pc2_variance": float(var[1]),
        "n_tasks": len(tasks),
        "n_att1_pass": sum(att1_passed),
        "n_att1_fail": sum(not p for p in att1_passed),
    }


# ── Analysis 13: Prompt-Length Correlation Check ──────────────────────────

def analysis_13_prompt_correlation(
    records: list[TrajectoryRecord],
    output_dir: Path,
) -> dict:
    """Check if PCA axes correlate with prompt_tokens (prompt-structure confound).

    Fits PCA on ALL attempts (same as Analysis 4), then computes Pearson
    correlation between PC1/PC2 and prompt_tokens. High |r| means that PC axis
    is mostly encoding prompt length — a proxy for attempt index.
    """
    _section("Analysis 13: Prompt-Length Correlation")

    all_vectors: list[np.ndarray] = []
    all_prompt_tokens: list[int] = []
    for rec in records:
        for att in rec.attempts:
            all_vectors.append(att.hidden_state)
            all_prompt_tokens.append(att.prompt_tokens)

    X = np.stack(all_vectors).astype(np.float32)
    prompt_tokens = np.array(all_prompt_tokens, dtype=np.float64)

    pca = PCA(n_components=2, random_state=SEED)
    X_2d = pca.fit_transform(X)

    var = pca.explained_variance_ratio_

    # Pearson correlation
    from scipy import stats
    r_pc1, p_pc1 = stats.pearsonr(X_2d[:, 0], prompt_tokens)
    r_pc2, p_pc2 = stats.pearsonr(X_2d[:, 1], prompt_tokens)

    print(f"  PCA variance: PC1={var[0]:.1%}  PC2={var[1]:.1%}")
    print(f"  Pearson r(PC1, prompt_tokens) = {r_pc1:.3f}  (p={p_pc1:.2e})")
    print(f"  Pearson r(PC2, prompt_tokens) = {r_pc2:.3f}  (p={p_pc2:.2e})")

    confound_pc1 = "YES - prompt structure" if abs(r_pc1) > 0.7 else "no"
    confound_pc2 = "YES - prompt structure" if abs(r_pc2) > 0.7 else "no"
    print(f"  PC1 confound: {confound_pc1}")
    print(f"  PC2 confound: {confound_pc2}")

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, pc_idx, r_val, p_val in [
        (axes[0], 0, r_pc1, p_pc1),
        (axes[1], 1, r_pc2, p_pc2),
    ]:
        ax.scatter(prompt_tokens, X_2d[:, pc_idx], s=8, alpha=0.3, c="#3498db")
        ax.set_xlabel("prompt_tokens")
        ax.set_ylabel(f"PC{pc_idx+1} ({var[pc_idx]:.1%} var)")
        ax.set_title(f"PC{pc_idx+1} vs prompt_tokens (r={r_val:.3f}, p={p_val:.1e})")

    plt.tight_layout()
    fig.savefig(output_dir / "pc_vs_prompt_tokens.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'pc_vs_prompt_tokens.png'}")

    return {
        "pc1_variance": float(var[0]),
        "pc2_variance": float(var[1]),
        "r_pc1_prompt": float(r_pc1),
        "p_pc1_prompt": float(p_pc1),
        "r_pc2_prompt": float(r_pc2),
        "p_pc2_prompt": float(p_pc2),
    }


# ── Analysis 14: Residualized Probe ─────────────────────────────────────────

def analysis_14_residualized_probe(
    records: list[TrajectoryRecord],
    seed: int,
) -> dict:
    """Train probe on hidden states with prompt_tokens regressed out.

    Compares three probes on attempt-0 data:
      1. Raw (unchanged) - baseline
      2. Residualized - prompt_tokens linearly removed from each dimension
      3. Prompt-only - just prompt_tokens as the single feature (floor)

    Residualization is fit on the train split only and applied to both sets.
    """
    _section("Analysis 14: Residualized Probe (prompt-length control)")

    # Collect attempt-0 data
    X_list, y_list, pt_list = [], [], []
    for rec in records:
        att0 = rec.attempts[0]
        X_list.append(att0.hidden_state)
        y_list.append(int(att0.passed))
        pt_list.append(att0.prompt_tokens)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    pt = np.array(pt_list, dtype=np.float32)

    # Train/test split (same as Analysis 1)
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, stratify=y, random_state=seed)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    pt_train, pt_test = pt[train_idx], pt[test_idx]

    # 1. Raw probe
    raw_result, _ = train_probe(
        X_train, y_train, X_test, y_test,
        pca_variance=0.95, cv_folds=5, seed=seed, layer_label="raw")
    print(f"  Raw probe:          AUC={raw_result.test_auc:.3f}  "
          f"CV={raw_result.cv_auc_mean:.3f}")

    # 2. Residualized probe
    reg = LinearRegression()
    reg.fit(pt_train.reshape(-1, 1), X_train)
    X_train_r = (X_train - reg.predict(pt_train.reshape(-1, 1))).astype(np.float32)
    X_test_r = (X_test - reg.predict(pt_test.reshape(-1, 1))).astype(np.float32)

    resid_result, _ = train_probe(
        X_train_r, y_train, X_test_r, y_test,
        pca_variance=0.95, cv_folds=5, seed=seed, layer_label="residualized")
    print(f"  Residualized probe: AUC={resid_result.test_auc:.3f}  "
          f"CV={resid_result.cv_auc_mean:.3f}")

    # 3. Prompt-only probe
    prompt_result, _ = train_probe(
        pt_train.reshape(-1, 1), y_train,
        pt_test.reshape(-1, 1), y_test,
        pca_variance=0.99, cv_folds=5, seed=seed, layer_label="prompt_only")
    print(f"  Prompt-only probe:  AUC={prompt_result.test_auc:.3f}  "
          f"CV={prompt_result.cv_auc_mean:.3f}")

    auc_drop = raw_result.test_auc - resid_result.test_auc
    print(f"  AUC drop from residualization: {auc_drop:+.3f}")
    print(f"  Signal above prompt-only: "
          f"{resid_result.test_auc - prompt_result.test_auc:+.3f}")

    return {
        "raw_auc": float(raw_result.test_auc),
        "raw_cv_auc": float(raw_result.cv_auc_mean),
        "residualized_auc": float(resid_result.test_auc),
        "residualized_cv_auc": float(resid_result.cv_auc_mean),
        "prompt_only_auc": float(prompt_result.test_auc),
        "prompt_only_cv_auc": float(prompt_result.cv_auc_mean),
        "auc_drop": float(auc_drop),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }


# ── Analysis 15: Direction Residualized Against delta_prompt_tokens ─────────

def analysis_15_direction_residualized(
    records: list[TrajectoryRecord],
    seed: int,
    output_dir: Path,
) -> dict:
    """Residualize 0->1 deltas against delta_prompt_tokens, then recompute direction.

    If the repair direction is just encoding prompt-token growth, residualization
    kills it. If real, it survives. Also runs a permutation test on residualized
    deltas.
    """
    _section("Analysis 15: Direction Residualized (prompt-length control)")

    # Collect 0->1 deltas WITH delta_prompt_tokens
    deltas, labels, delta_pts = [], [], []
    for rec in records:
        if rec.is_repetition_loop:
            continue
        if len(rec.attempts) < 2:
            continue
        curr, nxt = rec.attempts[0], rec.attempts[1]
        if curr.passed:
            continue
        if curr.completion_tokens == TOKEN_LIMIT:
            continue
        deltas.append(nxt.hidden_state - curr.hidden_state)
        labels.append(nxt.passed)
        delta_pts.append(nxt.prompt_tokens - curr.prompt_tokens)

    if not deltas:
        print("  SKIPPED (no 0->1 transitions)")
        return {}

    deltas_arr = np.stack(deltas).astype(np.float32)
    labels_arr = np.array(labels)
    delta_pts_arr = np.array(delta_pts, dtype=np.float32)

    n_succ = int(labels_arr.sum())
    n_fail = len(labels_arr) - n_succ
    print(f"  N_success={n_succ}  N_failure={n_fail}")

    # Raw direction norm (for comparison)
    mean_s = deltas_arr[labels_arr].mean(axis=0)
    mean_f = deltas_arr[~labels_arr].mean(axis=0)
    raw_norm = float(np.linalg.norm(mean_s - mean_f))
    print(f"  Raw direction norm: {raw_norm:.2f}")

    # Residualize each dimension against delta_prompt_tokens
    reg = LinearRegression()
    reg.fit(delta_pts_arr.reshape(-1, 1), deltas_arr)
    deltas_resid = (deltas_arr - reg.predict(delta_pts_arr.reshape(-1, 1))).astype(np.float32)

    mean_s_r = deltas_resid[labels_arr].mean(axis=0)
    mean_f_r = deltas_resid[~labels_arr].mean(axis=0)
    resid_norm = float(np.linalg.norm(mean_s_r - mean_f_r))
    print(f"  Residualized direction norm: {resid_norm:.2f}")
    print(f"  Norm retention: {resid_norm / raw_norm:.1%}")

    # Permutation test on residualized deltas
    rng = np.random.default_rng(seed)
    n_perms = 1000
    perm_norms = np.empty(n_perms)
    for i in range(n_perms):
        shuffled = rng.permutation(labels_arr)
        ms = deltas_resid[shuffled].mean(axis=0)
        mf = deltas_resid[~shuffled].mean(axis=0)
        perm_norms[i] = np.linalg.norm(ms - mf)

    p_value = float((perm_norms >= resid_norm).mean())
    print(f"  Permutation test (residualized): p={p_value:.4f}  "
          f"null mean={perm_norms.mean():.2f}  null std={perm_norms.std():.2f}")

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm_norms, bins=50, color="#95a5a6", alpha=0.7, label="Permuted (residualized)")
    ax.axvline(resid_norm, color="#e74c3c", linewidth=2,
               label=f"Real norm = {resid_norm:.2f}")
    ax.set_xlabel("||mean_success - mean_failure|| (residualized)")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation Test: Residualized 0->1 Direction (p={p_value:.4f})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "direction_residualized_permtest.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'direction_residualized_permtest.png'}")

    return {
        "raw_norm": raw_norm,
        "residualized_norm": resid_norm,
        "norm_retention": resid_norm / raw_norm,
        "p_value": p_value,
        "perm_mean": float(perm_norms.mean()),
        "perm_std": float(perm_norms.std()),
        "n_success": n_succ,
        "n_failure": n_fail,
        "n_permutations": n_perms,
    }


# ── Analysis 16: Split-Half Stability Test ──────────────────────────────────

def analysis_16_split_half_stability(
    records: list[TrajectoryRecord],
    seed: int,
    output_dir: Path,
) -> dict:
    """Test whether the repair direction's inclination is stable across data splits.

    Splits 0->1 deltas into halves A/B (stratified by label), computes the
    direction on each half, measures cosine similarity. Repeats 1000 times.
    Compares to a null distribution where labels are shuffled before splitting.
    """
    _section("Analysis 16: Split-Half Stability (direction inclination)")

    deltas, labels = _collect_transition_deltas(records, trans_idx=0)
    if not deltas:
        print("  SKIPPED (no 0->1 transitions)")
        return {}

    deltas_arr = np.stack(deltas).astype(np.float32)
    labels_arr = np.array(labels)

    n_succ = int(labels_arr.sum())
    n_fail = len(labels_arr) - n_succ
    print(f"  N_success={n_succ}  N_failure={n_fail}  total={len(labels_arr)}")

    if n_succ < 4 or n_fail < 4:
        print("  SKIPPED (need at least 4 in each class for split-half)")
        return {}

    n_splits = 1000

    def _compute_split_cosines(labs: np.ndarray) -> np.ndarray:
        splitter = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=0.5, random_state=seed)
        cosines = np.empty(n_splits)
        for i, (idx_a, idx_b) in enumerate(splitter.split(deltas_arr, labs)):
            d_a, l_a = deltas_arr[idx_a], labs[idx_a]
            d_b, l_b = deltas_arr[idx_b], labs[idx_b]
            dir_a = d_a[l_a].mean(0) - d_a[~l_a].mean(0)
            dir_b = d_b[l_b].mean(0) - d_b[~l_b].mean(0)
            cosines[i] = _cosine_sim(dir_a, dir_b)
        return cosines

    # Real cosines
    real_cosines = _compute_split_cosines(labels_arr)
    print(f"  Real split-half cosine: mean={real_cosines.mean():.3f}  "
          f"std={real_cosines.std():.3f}  "
          f"min={real_cosines.min():.3f}  max={real_cosines.max():.3f}")

    # Null cosines (shuffled labels)
    rng = np.random.default_rng(seed)
    null_cosines = np.empty(n_splits)
    for i in range(n_splits):
        shuffled = rng.permutation(labels_arr)
        splitter_null = StratifiedShuffleSplit(
            n_splits=1, test_size=0.5, random_state=seed + i + 1)
        idx_a, idx_b = next(splitter_null.split(deltas_arr, shuffled))
        d_a, l_a = deltas_arr[idx_a], shuffled[idx_a]
        d_b, l_b = deltas_arr[idx_b], shuffled[idx_b]
        dir_a = d_a[l_a].mean(0) - d_a[~l_a].mean(0)
        dir_b = d_b[l_b].mean(0) - d_b[~l_b].mean(0)
        null_cosines[i] = _cosine_sim(dir_a, dir_b)

    print(f"  Null split-half cosine: mean={null_cosines.mean():.3f}  "
          f"std={null_cosines.std():.3f}")

    # p-value: fraction of null cosines >= real mean
    p_value = float((null_cosines >= real_cosines.mean()).mean())
    print(f"  p-value (null >= real mean): {p_value:.4f}")

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(real_cosines, bins=50, color="#27ae60", alpha=0.6, label="Real splits")
    ax.hist(null_cosines, bins=50, color="#e74c3c", alpha=0.6, label="Shuffled labels")
    ax.axvline(real_cosines.mean(), color="#27ae60", linewidth=2, linestyle="--",
               label=f"Real mean = {real_cosines.mean():.3f}")
    ax.axvline(null_cosines.mean(), color="#e74c3c", linewidth=2, linestyle="--",
               label=f"Null mean = {null_cosines.mean():.3f}")
    ax.set_xlabel("Cosine similarity (direction_A, direction_B)")
    ax.set_ylabel("Count")
    ax.set_title(f"Split-Half Stability: 0->1 Direction (p={p_value:.4f})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "split_half_stability.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'split_half_stability.png'}")

    return {
        "real_cosine_mean": float(real_cosines.mean()),
        "real_cosine_std": float(real_cosines.std()),
        "real_cosine_min": float(real_cosines.min()),
        "real_cosine_max": float(real_cosines.max()),
        "null_cosine_mean": float(null_cosines.mean()),
        "null_cosine_std": float(null_cosines.std()),
        "p_value": p_value,
        "n_splits": n_splits,
        "n_success": n_succ,
        "n_failure": n_fail,
    }


# ── Analysis 17: Permutation Test in PCA-100 Space ─────────────────────────

def analysis_17_permutation_test_pca(
    records_pca: list[TrajectoryRecord],
    seed: int,
    output_dir: Path,
) -> dict:
    """Permutation test on 0->1 direction in PCA-reduced space.

    Same as Analysis 6 but on PCA-100 reduced deltas.
    """
    _section("Analysis 17: Permutation Test (0->1, PCA-100 space)")

    deltas, labels = _collect_transition_deltas(records_pca, trans_idx=0)
    if not deltas:
        print("  SKIPPED (no 0->1 transitions)")
        return {}

    deltas_arr = np.stack(deltas)
    labels_arr = np.array(labels)
    n_succ = int(labels_arr.sum())
    n_fail = len(labels_arr) - n_succ

    if n_succ == 0 or n_fail == 0:
        print("  SKIPPED (need both pass and fail labels)")
        return {}

    mean_succ = deltas_arr[labels_arr].mean(axis=0)
    mean_fail = deltas_arr[~labels_arr].mean(axis=0)
    real_norm = float(np.linalg.norm(mean_succ - mean_fail))
    print(f"  N_success={n_succ}  N_failure={n_fail}")
    print(f"  Real direction norm (PCA-100): {real_norm:.2f}")

    rng = np.random.default_rng(seed)
    n_perms = 1000
    perm_norms = np.empty(n_perms)
    for i in range(n_perms):
        shuffled = rng.permutation(labels_arr)
        ms = deltas_arr[shuffled].mean(axis=0)
        mf = deltas_arr[~shuffled].mean(axis=0)
        perm_norms[i] = np.linalg.norm(ms - mf)

    p_value = float((perm_norms >= real_norm).mean())
    print(f"  Permutation norms: mean={perm_norms.mean():.2f}  "
          f"std={perm_norms.std():.2f}  max={perm_norms.max():.2f}")
    print(f"  p-value: {p_value:.4f}  "
          f"({'significant' if p_value < 0.05 else 'NOT significant'} at alpha=0.05)")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm_norms, bins=50, color="#95a5a6", alpha=0.7, label="Permuted norms")
    ax.axvline(real_norm, color="#e74c3c", linewidth=2,
               label=f"Real norm = {real_norm:.2f}")
    ax.set_xlabel("||mean_success - mean_failure|| (PCA-100)")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation Test: 0->1 Direction in PCA-100 (n={len(deltas)}, "
                 f"p={p_value:.4f})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "permutation_test_pca_0to1.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'permutation_test_pca_0to1.png'}")

    return {
        "real_norm": real_norm,
        "perm_mean": float(perm_norms.mean()),
        "perm_std": float(perm_norms.std()),
        "perm_max": float(perm_norms.max()),
        "p_value": p_value,
        "n_permutations": n_perms,
        "n_success": n_succ,
        "n_failure": n_fail,
    }


# ── Analysis 18: Permutation Test in PCA-100 Space (Residualized) ───────────

def analysis_18_permutation_test_pca_residualized(
    records_pca: list[TrajectoryRecord],
    seed: int,
    output_dir: Path,
) -> dict:
    """Permutation test on 0->1 direction in PCA-100 space, residualized against delta_prompt_tokens.

    Same as Analysis 17 but with prompt-token growth regressed out first.
    """
    _section("Analysis 18: Permutation Test (0->1, PCA-100, residualized)")

    # Collect deltas + delta_prompt_tokens
    deltas, labels, delta_pts = [], [], []
    for rec in records_pca:
        if rec.is_repetition_loop:
            continue
        if len(rec.attempts) < 2:
            continue
        curr, nxt = rec.attempts[0], rec.attempts[1]
        if curr.passed:
            continue
        if curr.completion_tokens == TOKEN_LIMIT:
            continue
        deltas.append(nxt.hidden_state - curr.hidden_state)
        labels.append(nxt.passed)
        delta_pts.append(nxt.prompt_tokens - curr.prompt_tokens)

    if not deltas:
        print("  SKIPPED (no 0->1 transitions)")
        return {}

    deltas_arr = np.stack(deltas).astype(np.float32)
    labels_arr = np.array(labels)
    delta_pts_arr = np.array(delta_pts, dtype=np.float32)

    n_succ = int(labels_arr.sum())
    n_fail = len(labels_arr) - n_succ
    if n_succ == 0 or n_fail == 0:
        print("  SKIPPED (need both pass and fail labels)")
        return {}

    # Residualize
    reg = LinearRegression()
    reg.fit(delta_pts_arr.reshape(-1, 1), deltas_arr)
    deltas_resid = (deltas_arr - reg.predict(delta_pts_arr.reshape(-1, 1))).astype(np.float32)

    mean_s = deltas_resid[labels_arr].mean(axis=0)
    mean_f = deltas_resid[~labels_arr].mean(axis=0)
    real_norm = float(np.linalg.norm(mean_s - mean_f))
    print(f"  N_success={n_succ}  N_failure={n_fail}")
    print(f"  Residualized direction norm (PCA-100): {real_norm:.2f}")

    rng = np.random.default_rng(seed)
    n_perms = 1000
    perm_norms = np.empty(n_perms)
    for i in range(n_perms):
        shuffled = rng.permutation(labels_arr)
        ms = deltas_resid[shuffled].mean(axis=0)
        mf = deltas_resid[~shuffled].mean(axis=0)
        perm_norms[i] = np.linalg.norm(ms - mf)

    p_value = float((perm_norms >= real_norm).mean())
    print(f"  Permutation norms: mean={perm_norms.mean():.2f}  "
          f"std={perm_norms.std():.2f}  max={perm_norms.max():.2f}")
    print(f"  p-value: {p_value:.4f}  "
          f"({'significant' if p_value < 0.05 else 'NOT significant'} at alpha=0.05)")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm_norms, bins=50, color="#95a5a6", alpha=0.7, label="Permuted (residualized)")
    ax.axvline(real_norm, color="#e74c3c", linewidth=2,
               label=f"Real norm = {real_norm:.2f}")
    ax.set_xlabel("||mean_success - mean_failure|| (PCA-100, residualized)")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation Test: PCA-100 Residualized (p={p_value:.4f})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "permutation_test_pca_resid_0to1.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'permutation_test_pca_resid_0to1.png'}")

    return {
        "real_norm": real_norm,
        "perm_mean": float(perm_norms.mean()),
        "perm_std": float(perm_norms.std()),
        "perm_max": float(perm_norms.max()),
        "p_value": p_value,
        "n_permutations": n_perms,
        "n_success": n_succ,
        "n_failure": n_fail,
    }


# ── Analysis 19: Permutation Test with PCA on Deltas ───────────────────────

def analysis_19_permutation_test_pca_deltas(
    records: list[TrajectoryRecord],
    seed: int,
    output_dir: Path,
    n_components: int = 100,
) -> dict:
    """Permutation test on 0->1 direction with PCA fitted on the deltas themselves.

    Unlike Analyses 17/18 (PCA fitted on attempt-0 states), this fits PCA on
    the 0->1 deltas directly. This is the most natural space for studying
    deltas, but it is also somewhat circular: PCA will tend to preserve the
    dominant patterns in the deltas, which includes the repair direction itself.
    The permutation test still controls for this (null is in the same space),
    but the result should be interpreted with this caveat.

    Runs both raw and residualized (delta_prompt_tokens removed) variants.
    """
    _section("Analysis 19: Permutation Test (0->1, PCA on deltas)")

    # Collect deltas + metadata
    deltas, labels, delta_pts = [], [], []
    for rec in records:
        if rec.is_repetition_loop:
            continue
        if len(rec.attempts) < 2:
            continue
        curr, nxt = rec.attempts[0], rec.attempts[1]
        if curr.passed:
            continue
        if curr.completion_tokens == TOKEN_LIMIT:
            continue
        deltas.append(nxt.hidden_state - curr.hidden_state)
        labels.append(nxt.passed)
        delta_pts.append(nxt.prompt_tokens - curr.prompt_tokens)

    if not deltas:
        print("  SKIPPED (no 0->1 transitions)")
        return {}

    deltas_arr = np.stack(deltas).astype(np.float32)
    labels_arr = np.array(labels)
    delta_pts_arr = np.array(delta_pts, dtype=np.float32)

    n_succ = int(labels_arr.sum())
    n_fail = len(labels_arr) - n_succ
    if n_succ == 0 or n_fail == 0:
        print("  SKIPPED (need both pass and fail labels)")
        return {}

    # Fit PCA on the deltas
    n_comp = min(n_components, deltas_arr.shape[0], deltas_arr.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    deltas_pca = pca.fit_transform(deltas_arr)
    var_explained = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA fitted on {len(deltas_arr)} deltas -> {n_comp} components")
    print(f"  Variance explained: {var_explained:.1%}")
    print(f"  N_success={n_succ}  N_failure={n_fail}")

    rng = np.random.default_rng(seed)
    n_perms = 1000
    results = {}

    # --- Raw (no residualization) ---
    mean_s = deltas_pca[labels_arr].mean(axis=0)
    mean_f = deltas_pca[~labels_arr].mean(axis=0)
    real_norm = float(np.linalg.norm(mean_s - mean_f))
    print(f"  Raw direction norm (PCA-deltas): {real_norm:.2f}")

    perm_norms = np.empty(n_perms)
    for i in range(n_perms):
        shuffled = rng.permutation(labels_arr)
        ms = deltas_pca[shuffled].mean(axis=0)
        mf = deltas_pca[~shuffled].mean(axis=0)
        perm_norms[i] = np.linalg.norm(ms - mf)

    p_raw = float((perm_norms >= real_norm).mean())
    print(f"  Permutation: null mean={perm_norms.mean():.2f}  "
          f"std={perm_norms.std():.2f}  p={p_raw:.4f}")

    results["raw_norm"] = real_norm
    results["raw_perm_mean"] = float(perm_norms.mean())
    results["raw_perm_std"] = float(perm_norms.std())
    results["raw_p_value"] = p_raw

    # Plot raw
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(perm_norms, bins=50, color="#95a5a6", alpha=0.7, label="Permuted")
    ax.axvline(real_norm, color="#e74c3c", linewidth=2,
               label=f"Real = {real_norm:.2f}")
    ax.set_xlabel("||mean_success - mean_failure||")
    ax.set_ylabel("Count")
    ax.set_title(f"PCA-on-deltas, raw (p={p_raw:.4f})")
    ax.legend()

    # --- Residualized ---
    reg = LinearRegression()
    reg.fit(delta_pts_arr.reshape(-1, 1), deltas_pca)
    deltas_pca_resid = (deltas_pca - reg.predict(
        delta_pts_arr.reshape(-1, 1))).astype(np.float32)

    mean_s_r = deltas_pca_resid[labels_arr].mean(axis=0)
    mean_f_r = deltas_pca_resid[~labels_arr].mean(axis=0)
    resid_norm = float(np.linalg.norm(mean_s_r - mean_f_r))
    print(f"  Residualized direction norm (PCA-deltas): {resid_norm:.2f}")

    rng2 = np.random.default_rng(seed)
    perm_norms_r = np.empty(n_perms)
    for i in range(n_perms):
        shuffled = rng2.permutation(labels_arr)
        ms = deltas_pca_resid[shuffled].mean(axis=0)
        mf = deltas_pca_resid[~shuffled].mean(axis=0)
        perm_norms_r[i] = np.linalg.norm(ms - mf)

    p_resid = float((perm_norms_r >= resid_norm).mean())
    print(f"  Permutation (resid): null mean={perm_norms_r.mean():.2f}  "
          f"std={perm_norms_r.std():.2f}  p={p_resid:.4f}")

    results["resid_norm"] = resid_norm
    results["resid_perm_mean"] = float(perm_norms_r.mean())
    results["resid_perm_std"] = float(perm_norms_r.std())
    results["resid_p_value"] = p_resid

    # Plot residualized
    ax = axes[1]
    ax.hist(perm_norms_r, bins=50, color="#95a5a6", alpha=0.7, label="Permuted")
    ax.axvline(resid_norm, color="#e74c3c", linewidth=2,
               label=f"Real = {resid_norm:.2f}")
    ax.set_xlabel("||mean_success - mean_failure|| (residualized)")
    ax.set_ylabel("Count")
    ax.set_title(f"PCA-on-deltas, residualized (p={p_resid:.4f})")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "permutation_test_pca_deltas.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'permutation_test_pca_deltas.png'}")

    results["pca_variance_explained"] = var_explained
    results["n_components"] = n_comp
    results["n_permutations"] = n_perms
    results["n_success"] = n_succ
    results["n_failure"] = n_fail
    return results


# ── Analysis 20: Split-Half Stability on Residualized Deltas ────────────────

def analysis_20_split_half_residualized(
    records: list[TrajectoryRecord],
    seed: int,
    output_dir: Path,
) -> dict:
    """Split-half stability test on deltas after residualizing delta_prompt_tokens.

    Same as Analysis 16 but first removes the linear effect of prompt-token
    growth from each delta dimension. This tests whether the direction's
    inclination is stable beyond the trivially stable prompt-length component.
    """
    _section("Analysis 20: Split-Half Stability (residualized)")

    # Collect deltas + delta_prompt_tokens
    deltas, labels, delta_pts = [], [], []
    for rec in records:
        if rec.is_repetition_loop:
            continue
        if len(rec.attempts) < 2:
            continue
        curr, nxt = rec.attempts[0], rec.attempts[1]
        if curr.passed:
            continue
        if curr.completion_tokens == TOKEN_LIMIT:
            continue
        deltas.append(nxt.hidden_state - curr.hidden_state)
        labels.append(nxt.passed)
        delta_pts.append(nxt.prompt_tokens - curr.prompt_tokens)

    if not deltas:
        print("  SKIPPED (no 0->1 transitions)")
        return {}

    deltas_arr = np.stack(deltas).astype(np.float32)
    labels_arr = np.array(labels)
    delta_pts_arr = np.array(delta_pts, dtype=np.float32)

    n_succ = int(labels_arr.sum())
    n_fail = len(labels_arr) - n_succ
    print(f"  N_success={n_succ}  N_failure={n_fail}  total={len(labels_arr)}")

    if n_succ < 4 or n_fail < 4:
        print("  SKIPPED (need at least 4 in each class)")
        return {}

    # Residualize deltas against delta_prompt_tokens
    reg = LinearRegression()
    reg.fit(delta_pts_arr.reshape(-1, 1), deltas_arr)
    deltas_resid = (deltas_arr - reg.predict(
        delta_pts_arr.reshape(-1, 1))).astype(np.float32)
    print("  Residualized deltas against delta_prompt_tokens")

    n_splits = 1000

    def _compute_split_cosines(data: np.ndarray, labs: np.ndarray) -> np.ndarray:
        splitter = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=0.5, random_state=seed)
        cosines = np.empty(n_splits)
        for i, (idx_a, idx_b) in enumerate(splitter.split(data, labs)):
            d_a, l_a = data[idx_a], labs[idx_a]
            d_b, l_b = data[idx_b], labs[idx_b]
            dir_a = d_a[l_a].mean(0) - d_a[~l_a].mean(0)
            dir_b = d_b[l_b].mean(0) - d_b[~l_b].mean(0)
            cosines[i] = _cosine_sim(dir_a, dir_b)
        return cosines

    # Real cosines on residualized deltas
    real_cosines = _compute_split_cosines(deltas_resid, labels_arr)
    print(f"  Real split-half cosine: mean={real_cosines.mean():.3f}  "
          f"std={real_cosines.std():.3f}  "
          f"min={real_cosines.min():.3f}  max={real_cosines.max():.3f}")

    # Null cosines (shuffled labels on residualized deltas)
    rng = np.random.default_rng(seed)
    null_cosines = np.empty(n_splits)
    for i in range(n_splits):
        shuffled = rng.permutation(labels_arr)
        splitter_null = StratifiedShuffleSplit(
            n_splits=1, test_size=0.5, random_state=seed + i + 1)
        idx_a, idx_b = next(splitter_null.split(deltas_resid, shuffled))
        d_a, l_a = deltas_resid[idx_a], shuffled[idx_a]
        d_b, l_b = deltas_resid[idx_b], shuffled[idx_b]
        dir_a = d_a[l_a].mean(0) - d_a[~l_a].mean(0)
        dir_b = d_b[l_b].mean(0) - d_b[~l_b].mean(0)
        null_cosines[i] = _cosine_sim(dir_a, dir_b)

    print(f"  Null split-half cosine: mean={null_cosines.mean():.3f}  "
          f"std={null_cosines.std():.3f}")

    p_value = float((null_cosines >= real_cosines.mean()).mean())
    print(f"  p-value (null >= real mean): {p_value:.4f}")

    # Compare with raw Analysis 16
    print(f"  (Compare: raw Analysis 16 cosine was 0.932)")

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(real_cosines, bins=50, color="#27ae60", alpha=0.6, label="Real splits")
    ax.hist(null_cosines, bins=50, color="#e74c3c", alpha=0.6, label="Shuffled labels")
    ax.axvline(real_cosines.mean(), color="#27ae60", linewidth=2, linestyle="--",
               label=f"Real mean = {real_cosines.mean():.3f}")
    ax.axvline(null_cosines.mean(), color="#e74c3c", linewidth=2, linestyle="--",
               label=f"Null mean = {null_cosines.mean():.3f}")
    ax.set_xlabel("Cosine similarity (direction_A, direction_B)")
    ax.set_ylabel("Count")
    ax.set_title(f"Split-Half Stability: Residualized (p={p_value:.4f})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "split_half_stability_residualized.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'split_half_stability_residualized.png'}")

    return {
        "real_cosine_mean": float(real_cosines.mean()),
        "real_cosine_std": float(real_cosines.std()),
        "real_cosine_min": float(real_cosines.min()),
        "real_cosine_max": float(real_cosines.max()),
        "null_cosine_mean": float(null_cosines.mean()),
        "null_cosine_std": float(null_cosines.std()),
        "p_value": p_value,
        "n_splits": n_splits,
        "n_success": n_succ,
        "n_failure": n_fail,
    }


# ── Analysis 21: Repeated-Splits Probe ──────────────────────────────────────

def analysis_21_repeated_splits(
    records: list[TrajectoryRecord],
    n_splits: int = 50,
) -> dict:
    """Repeated-splits version of the residualized probe (paper-grade).

    Runs the full pipeline (raw / residualized / prompt-only) on n_splits
    random stratified 80/20 splits. Residualization is re-fit on each train
    fold (no leakage). Reports mean, std, 95% CI, min, max for each pipeline.
    """
    _section(f"Analysis 21: Repeated-Splits Probe (n={n_splits})")

    # Collect attempt-0 data once
    X_list, y_list, pt_list = [], [], []
    for rec in records:
        att0 = rec.attempts[0]
        X_list.append(att0.hidden_state)
        y_list.append(int(att0.passed))
        pt_list.append(att0.prompt_tokens)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    pt = np.array(pt_list, dtype=np.float32)

    print(f"  Total samples: {len(y)} (pass={y.sum()}, fail={(1-y).sum()})")

    # Per-split metrics for each pipeline. Each list ends up length n_splits.
    raw = {"test_auc": [], "cv_auc": [], "test_acc": [], "test_f1": []}
    resid = {"test_auc": [], "cv_auc": [], "test_acc": [], "test_f1": []}
    prompt = {"test_auc": [], "cv_auc": [], "test_acc": [], "test_f1": []}

    def _record(bucket: dict, result) -> None:
        bucket["test_auc"].append(float(result.test_auc))
        bucket["cv_auc"].append(float(result.cv_auc_mean))
        bucket["test_acc"].append(float(result.test_accuracy))
        bucket["test_f1"].append(float(result.test_f1))

    for seed in range(n_splits):
        # Stratified 80/20 split with this seed
        idx = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            idx, test_size=0.2, stratify=y, random_state=seed)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pt_train, pt_test = pt[train_idx], pt[test_idx]

        # 1. Raw probe
        raw_result, _ = train_probe(
            X_train, y_train, X_test, y_test,
            pca_variance=0.95, cv_folds=5, seed=seed, layer_label=f"raw_s{seed}")
        _record(raw, raw_result)

        # 2. Residualized probe — re-fit residualization on this train fold
        reg = LinearRegression()
        reg.fit(pt_train.reshape(-1, 1), X_train)
        X_train_r = (X_train - reg.predict(pt_train.reshape(-1, 1))).astype(np.float32)
        X_test_r = (X_test - reg.predict(pt_test.reshape(-1, 1))).astype(np.float32)
        resid_result, _ = train_probe(
            X_train_r, y_train, X_test_r, y_test,
            pca_variance=0.95, cv_folds=5, seed=seed, layer_label=f"resid_s{seed}")
        _record(resid, resid_result)

        # 3. Prompt-only baseline
        prompt_result, _ = train_probe(
            pt_train.reshape(-1, 1), y_train,
            pt_test.reshape(-1, 1), y_test,
            pca_variance=0.99, cv_folds=5, seed=seed, layer_label=f"po_s{seed}")
        _record(prompt, prompt_result)

        if (seed + 1) % 10 == 0:
            print(f"  Completed {seed+1}/{n_splits} splits", flush=True)

    def _summary(arr: list[float]) -> dict:
        a = np.array(arr)
        return {
            "mean": float(a.mean()),
            "std": float(a.std(ddof=1)),
            "ci95_half": float(1.96 * a.std(ddof=1) / np.sqrt(len(a))),
            "min": float(a.min()),
            "max": float(a.max()),
        }

    def _summarize_bucket(bucket: dict) -> dict:
        out = {k + "_summary": _summary(v) for k, v in bucket.items()}
        out.update({k + "_per_split": v for k, v in bucket.items()})
        return out

    raw_out = _summarize_bucket(raw)
    resid_out = _summarize_bucket(resid)
    prompt_out = _summarize_bucket(prompt)

    def _line(name: str, summary: dict, label: str) -> str:
        s = summary[label + "_summary"]
        return (f"  {name}  {label.replace('_', ' '):>10s}: "
                f"mean={s['mean']:.3f}  +/-{s['ci95_half']:.3f} (95% CI)  "
                f"std={s['std']:.3f}  min={s['min']:.3f}  max={s['max']:.3f}")

    for name, out in [("Raw           ", raw_out),
                      ("Residualized  ", resid_out),
                      ("Prompt-only   ", prompt_out)]:
        print(_line(name, out, "test_auc"))
        print(_line(name, out, "cv_auc"))
        print(_line(name, out, "test_acc"))
        print(_line(name, out, "test_f1"))

    return {
        "n_splits": n_splits,
        "raw": raw_out,
        "residualized": resid_out,
        "prompt_only": prompt_out,
    }


# ── Analysis 22: Conditional Sensitivity (covariate check + residualization) ─

def _collect_transition_deltas_with_covariates(
    records: list[TrajectoryRecord],
    trans_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Same filter as `_collect_transition_deltas` but also returns covariates.

    Returns (deltas, labels, covariates) where covariates is a dict with:
      - delta_prompt_tokens (float array, len N)
      - code_length_attempt_0 (float array, len N)
      - failure_type_attempt_0 (str array, len N)

    Filter logic must match `_collect_transition_deltas` exactly to keep the
    same 236 clean tasks (128/108).
    """
    deltas: list[np.ndarray] = []
    labels: list[bool] = []
    delta_pts: list[float] = []
    code_len_0: list[float] = []
    failure_types_0: list[str] = []

    for rec in records:
        if rec.is_repetition_loop:
            continue
        if len(rec.attempts) <= trans_idx + 1:
            continue
        curr = rec.attempts[trans_idx]
        nxt = rec.attempts[trans_idx + 1]
        if curr.passed:
            continue
        if curr.completion_tokens == TOKEN_LIMIT:
            continue
        deltas.append(nxt.hidden_state - curr.hidden_state)
        labels.append(nxt.passed)
        delta_pts.append(float(nxt.prompt_tokens - curr.prompt_tokens))
        code_len_0.append(float(curr.completion_tokens))
        failure_types_0.append(curr.failure_type)

    deltas_arr = np.stack(deltas).astype(np.float32)
    labels_arr = np.array(labels)
    covariates = {
        "delta_prompt_tokens": np.array(delta_pts, dtype=np.float32),
        "code_length_attempt_0": np.array(code_len_0, dtype=np.float32),
        "failure_type_attempt_0": np.array(failure_types_0),
    }
    return deltas_arr, labels_arr, covariates


def _build_covariate_matrix(
    covariates_to_use: dict,
    failure_types: np.ndarray | None = None,
) -> np.ndarray:
    """Stack continuous covariates and one-hot encode failure_type if present.

    Drops one failure_type level as baseline (the most frequent).
    """
    cols = []
    for v in covariates_to_use.values():
        if v.ndim == 1:
            cols.append(v.reshape(-1, 1))
        else:
            cols.append(v)
    if failure_types is not None:
        # one-hot, drop most frequent level as baseline
        unique, counts = np.unique(failure_types, return_counts=True)
        baseline = unique[np.argmax(counts)]
        for level in unique:
            if level == baseline:
                continue
            cols.append((failure_types == level).astype(np.float32).reshape(-1, 1))
    return np.hstack(cols).astype(np.float32)


def analysis_22_conditional_sensitivity(
    records: list[TrajectoryRecord],
    seed: int,
    output_dir: Path,
    alpha: float = 0.05,
) -> dict:
    """Sensitivity analysis: covariate-distribution check + conditional residualization.

    Step 22a: For each candidate covariate, test whether its distribution
    differs between the success group and the failure group on the 236 clean
    0->1 transitions. Welch's t-test for continuous, chi-squared for categorical.

    Step 22b: Residualize ONLY against covariates flagged "differs" at alpha.
    Rationale: the contrastive construction mean(success) - mean(failure)
    already cancels equally-distributed covariates. Residualizing against a
    covariate that does not differ between groups risks double-correcting
    (collapses signal artifactually).

    Step 22c: Recompute direction norm + permutation + split-half on
    residualized deltas (or raw if no covariates differ).
    """
    _section("Analysis 22: Conditional Sensitivity (covariate check + cond. residualization)")

    deltas, labels, covs = _collect_transition_deltas_with_covariates(records, trans_idx=0)
    n_succ = int(labels.sum())
    n_fail = len(labels) - n_succ
    print(f"  Clean 0->1 deltas: N={len(labels)}  success={n_succ}  failure={n_fail}")

    # ---- Step 22a: covariate-distribution check ----
    print("\n  Step 22a — Covariate-distribution check (success vs failure groups):")

    cov_results: dict = {}

    # delta_prompt_tokens
    s_vals = covs["delta_prompt_tokens"][labels]
    f_vals = covs["delta_prompt_tokens"][~labels]
    t_stat, p_val = stats.ttest_ind(s_vals, f_vals, equal_var=False)
    cov_results["delta_prompt_tokens"] = {
        "test": "welch_t",
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "mean_success": float(s_vals.mean()),
        "mean_failure": float(f_vals.mean()),
        "std_success": float(s_vals.std(ddof=1)),
        "std_failure": float(f_vals.std(ddof=1)),
        "differs": bool(p_val < alpha),
    }
    print(f"    delta_prompt_tokens: "
          f"mean_succ={s_vals.mean():.1f}  mean_fail={f_vals.mean():.1f}  "
          f"t={t_stat:.2f}  p={p_val:.4f}  "
          f"differs={cov_results['delta_prompt_tokens']['differs']}")

    # code_length_attempt_0
    s_vals = covs["code_length_attempt_0"][labels]
    f_vals = covs["code_length_attempt_0"][~labels]
    t_stat, p_val = stats.ttest_ind(s_vals, f_vals, equal_var=False)
    cov_results["code_length_attempt_0"] = {
        "test": "welch_t",
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "mean_success": float(s_vals.mean()),
        "mean_failure": float(f_vals.mean()),
        "std_success": float(s_vals.std(ddof=1)),
        "std_failure": float(f_vals.std(ddof=1)),
        "differs": bool(p_val < alpha),
    }
    print(f"    code_length_attempt_0: "
          f"mean_succ={s_vals.mean():.1f}  mean_fail={f_vals.mean():.1f}  "
          f"t={t_stat:.2f}  p={p_val:.4f}  "
          f"differs={cov_results['code_length_attempt_0']['differs']}")

    # failure_type (chi-squared on 4x2 contingency)
    ft = covs["failure_type_attempt_0"]
    levels = np.unique(ft)
    contingency = np.array([
        [int(((ft == lv) & labels).sum()), int(((ft == lv) & ~labels).sum())]
        for lv in levels
    ])
    chi2, p_val_chi, dof, expected = stats.chi2_contingency(contingency)
    cov_results["failure_type_attempt_0"] = {
        "test": "chi2",
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p_val_chi),
        "levels": levels.tolist(),
        "contingency_success": contingency[:, 0].tolist(),
        "contingency_failure": contingency[:, 1].tolist(),
        "differs": bool(p_val_chi < alpha),
    }
    print(f"    failure_type_attempt_0: levels={levels.tolist()}  "
          f"chi2={chi2:.2f}  dof={dof}  p={p_val_chi:.4f}  "
          f"differs={cov_results['failure_type_attempt_0']['differs']}")

    # ---- Step 22b: build conditional covariate matrix ----
    print("\n  Step 22b — Building conditional covariate matrix:")
    continuous_to_use = {}
    if cov_results["delta_prompt_tokens"]["differs"]:
        continuous_to_use["delta_prompt_tokens"] = covs["delta_prompt_tokens"]
        print("    INCLUDE: delta_prompt_tokens")
    else:
        print("    EXCLUDE: delta_prompt_tokens (no group difference)")
    if cov_results["code_length_attempt_0"]["differs"]:
        continuous_to_use["code_length_attempt_0"] = covs["code_length_attempt_0"]
        print("    INCLUDE: code_length_attempt_0")
    else:
        print("    EXCLUDE: code_length_attempt_0 (no group difference)")
    use_failure_type = cov_results["failure_type_attempt_0"]["differs"]
    if use_failure_type:
        print("    INCLUDE: failure_type_attempt_0 (one-hot, baseline = most frequent)")
    else:
        print("    EXCLUDE: failure_type_attempt_0 (no group difference)")

    n_covariates = len(continuous_to_use) + (
        len(np.unique(covs["failure_type_attempt_0"])) - 1 if use_failure_type else 0
    )
    print(f"    Total covariate columns: {n_covariates}")

    # ---- Step 22c: residualize (or skip) and recompute ----
    if n_covariates == 0:
        print("\n  Step 22c — No covariates differ: residualization is unwarranted.")
        print("    The raw direction (Analysis 6) stands as the result.")
        return {
            "covariate_check": cov_results,
            "covariates_used": [],
            "n_covariate_columns": 0,
            "residualization_performed": False,
            "n_success": n_succ,
            "n_failure": n_fail,
        }

    print("\n  Step 22c — Per-dimension OLS, subtract, recompute direction stats")
    cov_matrix = _build_covariate_matrix(
        continuous_to_use,
        failure_types=covs["failure_type_attempt_0"] if use_failure_type else None,
    )
    print(f"    Covariate matrix shape: {cov_matrix.shape}")

    # Raw norm for comparison
    mean_s_raw = deltas[labels].mean(axis=0)
    mean_f_raw = deltas[~labels].mean(axis=0)
    raw_norm = float(np.linalg.norm(mean_s_raw - mean_f_raw))
    print(f"    Raw direction norm: {raw_norm:.2f}")

    # Per-dimension OLS, subtract
    reg = LinearRegression()
    reg.fit(cov_matrix, deltas)
    deltas_resid = (deltas - reg.predict(cov_matrix)).astype(np.float32)

    mean_s = deltas_resid[labels].mean(axis=0)
    mean_f = deltas_resid[~labels].mean(axis=0)
    resid_norm = float(np.linalg.norm(mean_s - mean_f))
    print(f"    Residualized direction norm: {resid_norm:.2f}  "
          f"(retention {resid_norm/raw_norm:.1%})")

    # Permutation test on residualized deltas
    rng = np.random.default_rng(seed)
    n_perms = 1000
    perm_norms = np.empty(n_perms)
    for i in range(n_perms):
        shuffled = rng.permutation(labels)
        ms = deltas_resid[shuffled].mean(axis=0)
        mf = deltas_resid[~shuffled].mean(axis=0)
        perm_norms[i] = np.linalg.norm(ms - mf)
    p_norm = float((perm_norms >= resid_norm).mean())
    print(f"    Permutation test (residualized): "
          f"p={p_norm:.4f}  null mean={perm_norms.mean():.2f}  "
          f"null std={perm_norms.std():.2f}  null max={perm_norms.max():.2f}")

    # Split-half stability on residualized deltas
    n_splits_sh = 1000
    sss = StratifiedShuffleSplit(n_splits=n_splits_sh, test_size=0.5, random_state=seed)
    real_cosines = np.empty(n_splits_sh)
    for i, (a_idx, b_idx) in enumerate(sss.split(np.zeros(len(labels)), labels.astype(int))):
        a_lab, b_lab = labels[a_idx], labels[b_idx]
        v_a = deltas_resid[a_idx][a_lab].mean(axis=0) - deltas_resid[a_idx][~a_lab].mean(axis=0)
        v_b = deltas_resid[b_idx][b_lab].mean(axis=0) - deltas_resid[b_idx][~b_lab].mean(axis=0)
        real_cosines[i] = float(np.dot(v_a, v_b) / (np.linalg.norm(v_a) * np.linalg.norm(v_b)))

    # Null: shuffle labels, then split
    rng = np.random.default_rng(seed + 1)
    null_cosines = np.empty(n_splits_sh)
    sss_null = StratifiedShuffleSplit(n_splits=n_splits_sh, test_size=0.5, random_state=seed+2)
    shuffled_labels = rng.permutation(labels)
    for i, (a_idx, b_idx) in enumerate(sss_null.split(np.zeros(len(labels)), shuffled_labels.astype(int))):
        a_lab = shuffled_labels[a_idx]
        b_lab = shuffled_labels[b_idx]
        v_a = deltas_resid[a_idx][a_lab].mean(axis=0) - deltas_resid[a_idx][~a_lab].mean(axis=0)
        v_b = deltas_resid[b_idx][b_lab].mean(axis=0) - deltas_resid[b_idx][~b_lab].mean(axis=0)
        null_cosines[i] = float(np.dot(v_a, v_b) / (np.linalg.norm(v_a) * np.linalg.norm(v_b)))

    p_cos = float((null_cosines >= real_cosines.mean()).mean())
    print(f"    Split-half cosine (residualized): "
          f"mean={real_cosines.mean():.3f}  std={real_cosines.std():.3f}  "
          f"min={real_cosines.min():.3f}  max={real_cosines.max():.3f}  "
          f"null mean={null_cosines.mean():.3f}  p={p_cos:.4f}")

    # Histogram of permutation test
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm_norms, bins=50, color="#95a5a6", alpha=0.7,
            label="Permuted (residualized)")
    ax.axvline(resid_norm, color="#e74c3c", linewidth=2,
               label=f"Real norm = {resid_norm:.2f}")
    ax.set_xlabel("||mean_success - mean_failure|| (residualized, conditional)")
    ax.set_ylabel("Count")
    ax.set_title(f"Analysis 22: Conditional Sensitivity (p={p_norm:.4f})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "analysis22_conditional_permtest.png", dpi=150)
    plt.close(fig)
    print(f"    Saved: {output_dir / 'analysis22_conditional_permtest.png'}")

    return {
        "covariate_check": cov_results,
        "covariates_used": list(continuous_to_use.keys()) +
                          (["failure_type_attempt_0"] if use_failure_type else []),
        "n_covariate_columns": int(cov_matrix.shape[1]),
        "residualization_performed": True,
        "raw_norm": raw_norm,
        "residualized_norm": resid_norm,
        "norm_retention": resid_norm / raw_norm,
        "permutation": {
            "p_value": p_norm,
            "null_mean": float(perm_norms.mean()),
            "null_std": float(perm_norms.std()),
            "null_max": float(perm_norms.max()),
            "n_permutations": n_perms,
        },
        "split_half": {
            "real_cosine_mean": float(real_cosines.mean()),
            "real_cosine_std": float(real_cosines.std()),
            "real_cosine_min": float(real_cosines.min()),
            "real_cosine_max": float(real_cosines.max()),
            "null_cosine_mean": float(null_cosines.mean()),
            "null_cosine_std": float(null_cosines.std()),
            "p_value": p_cos,
            "n_splits": n_splits_sh,
        },
        "n_success": n_succ,
        "n_failure": n_fail,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def _make_serializable(obj: object) -> object:
    """Convert numpy types to native Python types for JSON serialization.

    Also skips keys starting with "_" (used for internal data like the
    repair_direction vector that shouldn't be written to JSON).
    """
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()
                if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print(f"Loading trajectories from: {RUN_DIR}")
    records = load_trajectories(RUN_DIR, LAYER)
    print(f"  Loaded {len(records)} tasks")

    # Tag (but don't remove) repetition-loop tasks
    n_loops = sum(1 for r in records if r.is_repetition_loop)
    print(f"  Tagged {n_loops} repetition-loop tasks "
          f"({n_loops}/{len(records)}) — kept for Analysis 1 & 4, "
          f"excluded from 2/3/5")

    print_data_summary(records)

    all_metrics: dict = {}

    # ---- Analysis 1: classifier uses ALL records (including loops) ----
    all_metrics["classifier"] = analysis_1_classifier(records, SEED)

    # ---- Analysis 2: direction excludes loops internally ----
    dir_metrics = analysis_2_direction(records)
    # Extract the repair direction vector (numpy array, not JSON-serializable)
    # and pass it to Analysis 3. The "_" prefix ensures it's skipped in JSON.
    repair_direction = dir_metrics.pop("_repair_direction", np.zeros(1))
    all_metrics["direction"] = dir_metrics

    # ---- Analysis 3: convergence excludes loops internally ----
    all_metrics["convergence"] = analysis_3_convergence(records, repair_direction)

    # ---- Analysis 4: PCA visualization uses ALL records (loops in gray) ----
    all_metrics["pca"] = analysis_4_pca(records, OUTPUT_DIR)

    # ---- Analysis 5: distance separates loops into their own curve ----
    all_metrics["distance"] = analysis_5_distance(records, OUTPUT_DIR)

    # ---- Analysis 6: permutation test on 0->1 direction ----
    all_metrics["permutation_test"] = analysis_6_permutation_test(
        records, SEED, OUTPUT_DIR)

    # ---- PCA reduction: fit on attempt-0, transform all (foundation for 8-10) ----
    records_pca, _pca_obj = _build_pca_reduced_records(records, n_components=100)

    # ---- Analysis 8: direction in PCA-reduced space ----
    dir_pca_metrics = analysis_8_direction_pca(records_pca)
    repair_dir_pca = dir_pca_metrics.pop(
        "_repair_direction",
        np.zeros(records_pca[0].attempts[0].hidden_state.shape[0]),
    )
    all_metrics["direction_pca"] = dir_pca_metrics

    # ---- Analysis 9: convergence in PCA-reduced space ----
    all_metrics["convergence_pca"] = analysis_9_convergence_pca(
        records_pca, repair_dir_pca)

    # ---- Analysis 10: distance with first-pass centroid in PCA space ----
    all_metrics["distance_pca"] = analysis_10_distance_pca(
        records_pca, OUTPUT_DIR)

    # ---- Analysis 11: projection histogram (0->1 deltas) ----
    all_metrics["projection_histogram"] = analysis_11_projection_histogram(
        records, repair_direction, OUTPUT_DIR)

    # ---- Analysis 12: focused PCA (attempt 0 vs 1) ----
    all_metrics["focused_pca"] = analysis_12_focused_pca(records, OUTPUT_DIR)

    # ---- Analysis 13: prompt-length correlation check ----
    all_metrics["prompt_correlation"] = analysis_13_prompt_correlation(
        records, OUTPUT_DIR)

    # ---- Analysis 14: residualized probe ----
    all_metrics["residualized_probe"] = analysis_14_residualized_probe(
        records, SEED)

    # ---- Analysis 15: direction residualized against delta_prompt_tokens ----
    all_metrics["direction_residualized"] = analysis_15_direction_residualized(
        records, SEED, OUTPUT_DIR)

    # ---- Analysis 16: split-half stability ----
    all_metrics["split_half_stability"] = analysis_16_split_half_stability(
        records, SEED, OUTPUT_DIR)

    # ---- Analysis 17: permutation test in PCA space ----
    all_metrics["permutation_test_pca"] = analysis_17_permutation_test_pca(
        records_pca, SEED, OUTPUT_DIR)

    # ---- Analysis 18: permutation test in PCA space (residualized) ----
    all_metrics["permutation_test_pca_resid"] = analysis_18_permutation_test_pca_residualized(
        records_pca, SEED, OUTPUT_DIR)

    # ---- Analysis 19: permutation test with PCA on deltas ----
    all_metrics["permutation_test_pca_deltas"] = analysis_19_permutation_test_pca_deltas(
        records, SEED, OUTPUT_DIR)

    # ---- Analysis 20: split-half stability (residualized) ----
    all_metrics["split_half_residualized"] = analysis_20_split_half_residualized(
        records, SEED, OUTPUT_DIR)

    # ---- Analysis 21: repeated-splits probe (paper-grade Result 1) ----
    all_metrics["repeated_splits_probe"] = analysis_21_repeated_splits(
        records, n_splits=50)

    # ---- Analysis 22: conditional sensitivity (covariate check + cond. resid.) ----
    all_metrics["analysis_22_sensitivity_v2"] = analysis_22_conditional_sensitivity(
        records, SEED, OUTPUT_DIR)

    # ---- Save all numerical metrics to JSON ----
    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_make_serializable(all_metrics), f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
