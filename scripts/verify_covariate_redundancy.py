"""Verify the two redundancy claims in the Result 2 "Note" of the README/paper.

The conditional residualization uses a 4-column covariate matrix in which the two
continuous covariates, `delta_prompt_tokens` and `code_length_attempt_0`, overlap:
the failed attempt-0 code is pasted into the repair prompt, so its length is part
of both. This script measures, on the exact 236 clean 0->1 transitions used by the
sensitivity check, the two numbers stated in the Note:

  1. Pearson r between `delta_prompt_tokens` and `code_length_attempt_0`.
  2. The increase in in-sample R^2 of the per-dimension OLS fit on the 236 deltas
     when the redundant `delta_prompt_tokens` column is added on top of the other
     three columns (`code_length_attempt_0` + the two failure-type dummies).

For (2), "in-sample R^2 of the OLS fit on the 236 deltas" is averaged across all
2560 hidden-state delta dimensions, since each dimension is residualized by its own
OLS. The reported gain is the mean R^2(full 4-col) - mean R^2(reduced 3-col).

Reuses the loaders and filters of `analyze_trajectories.py` so the 236-task sample
(128 success / 108 failure) matches the rest of the analysis exactly.

Run:  python scripts/verify_covariate_redundancy.py
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from analyze_trajectories import (
    RUN_DIR,
    LAYER,
    load_trajectories,
    _collect_transition_deltas_with_covariates,
    _build_covariate_matrix,
)


def _mean_in_sample_r2(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean in-sample R^2 of OLS (with intercept) fit per response column of Y.

    X: (n, p) predictors. Y: (n, d) responses (the 2560 delta dimensions).
    Fits one OLS per column of Y, returns the mean R^2 across columns.
    """
    n = X.shape[0]
    Xd = np.hstack([np.ones((n, 1), dtype=np.float64), X.astype(np.float64)])
    # Least squares solution for all response columns at once.
    beta, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
    Y_hat = Xd @ beta
    resid = Y - Y_hat
    ss_res = (resid ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    # Guard against zero-variance dimensions.
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
    return float(np.mean(r2))


def main() -> None:
    print(f"Loading trajectories from: {RUN_DIR}  (layer {LAYER})")
    records = load_trajectories(RUN_DIR, LAYER)
    print(f"  Loaded {len(records)} tasks")

    deltas, labels, covs = _collect_transition_deltas_with_covariates(records, trans_idx=0)
    n = len(labels)
    print(f"  Clean 0->1 deltas: N={n}  success={int(labels.sum())}  failure={n - int(labels.sum())}")
    assert n == 236, f"expected 236 clean transitions, got {n}"

    dpt = covs["delta_prompt_tokens"].astype(np.float64)
    cl0 = covs["code_length_attempt_0"].astype(np.float64)
    Y = deltas.astype(np.float64)  # (236, 2560)

    # ---- Claim 1: Pearson r between the two continuous covariates ----
    r, p = stats.pearsonr(dpt, cl0)
    print("\n[1] Pearson r(delta_prompt_tokens, code_length_attempt_0)")
    print(f"    r = {r:.4f}   (p = {p:.2e})")

    # ---- Claim 2: in-sample R^2 gain from adding the redundant column ----
    # Full 4-column matrix: dpt + cl0 + 2 failure-type dummies.
    full = _build_covariate_matrix(
        {"delta_prompt_tokens": dpt.astype(np.float32),
         "code_length_attempt_0": cl0.astype(np.float32)},
        failure_types=covs["failure_type_attempt_0"],
    )
    # Reduced 3-column matrix: drop the redundant delta_prompt_tokens column.
    reduced = _build_covariate_matrix(
        {"code_length_attempt_0": cl0.astype(np.float32)},
        failure_types=covs["failure_type_attempt_0"],
    )
    print(f"\n    Full matrix columns:    {full.shape[1]}")
    print(f"    Reduced matrix columns: {reduced.shape[1]}")

    r2_full = _mean_in_sample_r2(full, Y)
    r2_reduced = _mean_in_sample_r2(reduced, Y)
    gain_pp = (r2_full - r2_reduced) * 100.0
    ratio = full.shape[1] / n

    print("\n[2] Mean in-sample R^2 across the 2560 delta dimensions")
    print(f"    full (4 cols):    {r2_full:.5f}")
    print(f"    reduced (3 cols): {r2_reduced:.5f}")
    print(f"    gain from adding delta_prompt_tokens: {gain_pp:.4f} percentage points")
    print(f"\n    parameter-to-data ratio: {full.shape[1]}/{n} = {ratio:.4f} ({ratio*100:.1f}%)")

    print("\nSummary for the Note:")
    print(f"  Pearson r = {r:.3f}")
    print(f"  R^2 gain  = {gain_pp:.3f} pp  ({'under' if gain_pp < 1 else 'OVER'} one percentage point)")
    print(f"  ratio     = {ratio*100:.1f}%")


if __name__ == "__main__":
    main()
