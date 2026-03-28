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
from sklearn.model_selection import train_test_split

from reasoninglab.probing.data import _task_id_from_filename
from reasoninglab.probing.probe import train_probe


# ── Configuration ────────────────────────────────────────────────────────────
# Hardcoded paths and parameters. Change these when running on different data.

RUN_DIR = Path("runs/h2-trajectory-repair_b_20260328_123657")
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

    # ---- Save all numerical metrics to JSON ----
    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_make_serializable(all_metrics), f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
