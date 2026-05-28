#!/usr/bin/env python3
"""Prompt-only baseline with the SAME nested-CV-with-explicit-C-grid mechanism
as layer_nested_cv.py, for methodological parity.

No layer axis (the feature is the scalar prompt_tokens). No PCA (1 feature).
Inner loop selects C from the explicit grid by 5-fold CV on outer-train;
final refit at chosen C, evaluate once on outer-test. 50 outer splits.

This replaces the old GridSearchCV-path prompt-only number so all three probes
(raw / residualized / prompt-only) use the identical explicit-grid procedure.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from reasoninglab.probing.data import load_samples

DEFAULT_C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


def _fit_eval(X_train, y_train, X_val, y_val, *, C, seed) -> float:
    """StandardScaler -> LogReg(C) on the single prompt_tokens feature."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_va = scaler.transform(X_val)
    clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=seed)
    clf.fit(X_tr, y_train)
    proba = clf.predict_proba(X_va)[:, 1]
    return float(roc_auc_score(y_val, proba))


def _summary(arr):
    a = np.array(arr)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)),
        "ci95_half": float(1.96 * a.std(ddof=1) / np.sqrt(len(a))),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("hidden_states_dir")
    p.add_argument("--output-dir", default="results/h2/prompt_only_nested_cv")
    p.add_argument("--n-outer-splits", type=int, default=50)
    p.add_argument("--n-inner-folds", type=int, default=5)
    p.add_argument("--c-grid", type=float, nargs="*", default=DEFAULT_C_GRID)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prompt-only] Loading samples from {args.hidden_states_dir} ...")
    samples = load_samples(Path(args.hidden_states_dir))
    y = np.array([int(s.passed) for s in samples], dtype=np.int32)
    pt = np.array([s.prompt_tokens for s in samples], dtype=np.float32).reshape(-1, 1)
    c_grid = list(args.c_grid)
    print(f"[prompt-only] {len(samples)} samples, C grid: {c_grid}, "
          f"n_outer={args.n_outer_splits}, n_inner={args.n_inner_folds}")

    outer_test_aucs = []
    selected_Cs = []

    for outer_seed in range(args.n_outer_splits):
        idx = np.arange(len(y))
        ot_idx, ote_idx = train_test_split(
            idx, test_size=0.2, stratify=y, random_state=outer_seed)
        pt_ot, pt_ote = pt[ot_idx], pt[ote_idx]
        y_ot, y_ote = y[ot_idx], y[ote_idx]

        # inner CV over C
        skf = StratifiedKFold(n_splits=args.n_inner_folds, shuffle=True,
                              random_state=outer_seed)
        c_aucs = {C: [] for C in c_grid}
        for it_idx, iv_idx in skf.split(np.zeros(len(y_ot)), y_ot):
            for C in c_grid:
                auc = _fit_eval(pt_ot[it_idx], y_ot[it_idx],
                                pt_ot[iv_idx], y_ot[iv_idx], C=C, seed=outer_seed)
                c_aucs[C].append(auc)
        c_means = {C: float(np.mean(v)) for C, v in c_aucs.items()}
        best_C = max(c_means, key=c_means.get)
        selected_Cs.append(best_C)

        outer_auc = _fit_eval(pt_ot, y_ot, pt_ote, y_ote, C=best_C, seed=outer_seed)
        outer_test_aucs.append(outer_auc)

        if (outer_seed + 1) % 10 == 0:
            print(f"[prompt-only] outer {outer_seed+1}/{args.n_outer_splits}  "
                  f"C={best_C}  auc={outer_auc:.3f}", flush=True)

    s = _summary(outer_test_aucs)
    c_counts = Counter(selected_Cs)
    print("\n" + "=" * 60)
    print(f"PROMPT-ONLY NESTED CV  n_outer={args.n_outer_splits}")
    print("=" * 60)
    print(f"Outer test AUC: {s['mean']:.4f} +/- {s['ci95_half']:.4f} (95% CI)")
    print(f"                std={s['std']:.4f}  min={s['min']:.4f}  max={s['max']:.4f}")
    print(f"C-selection: {dict(c_counts)}")
    print("=" * 60)

    payload = {
        "n_outer_splits": args.n_outer_splits,
        "n_inner_folds": args.n_inner_folds,
        "c_grid": c_grid,
        "n_samples": len(samples),
        "outer_test_aucs": outer_test_aucs,
        "outer_test_auc_summary": s,
        "selected_Cs": selected_Cs,
        "C_selection_counts": {str(C): c_counts.get(C, 0) for C in c_grid},
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[prompt-only] Saved {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
