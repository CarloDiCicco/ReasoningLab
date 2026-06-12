#!/usr/bin/env python3
"""Prompt-only baseline, several model families, same nested-CV procedure.

Question this answers: is the linear probe's pass/fail signal a real internal
signal, or could prompt length alone explain it -- even nonlinearly? To check,
we predict pass/fail from prompt_tokens ALONE with several models (logistic,
random forest, gradient boosting, small MLP) under the IDENTICAL nested-CV
protocol used for the main probe: 50 outer stratified 80/20 splits, inner
5-fold CV grid search for hyperparameters, mean outer-test AUC +/- 95% CI.

If even a flexible nonlinear length model stays well below the residualized
probe (0.911) and near the linear length baseline (0.754), then the probe's
signal is not an artifact of a nonlinear length-difficulty relationship.

One feature only (prompt_tokens), 444 samples. StandardScaler is applied (a
no-op-ish but harmless single-feature scaling); no PCA (1 feature).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from reasoninglab.probing.data import load_samples


# ── Model zoo: (name, estimator, param grid) ─────────────────────────────────
def _model_specs(seed: int):
    return {
        "logistic": (
            LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed),
            {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=seed, n_jobs=1),
            {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [2, 3, 5, None],
                "clf__min_samples_leaf": [1, 5, 20],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=seed),
            {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [1, 2, 3],
                "clf__learning_rate": [0.03, 0.1],
            },
        ),
        "mlp": (
            MLPClassifier(max_iter=2000, random_state=seed),
            {
                "clf__hidden_layer_sizes": [(8,), (16,), (16, 8)],
                "clf__alpha": [1e-4, 1e-2, 1e-1],
            },
        ),
    }


def _fit_eval_one(name, X_tr, y_tr, X_te, y_te, *, cv_folds, seed) -> tuple[float, dict]:
    """Inner-CV grid search on train, eval AUC on test. Returns (auc, best_params)."""
    est, grid = _model_specs(seed)[name]
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", est)])
    inner = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    gs = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=1, refit=True)
    gs.fit(X_tr, y_tr)
    proba = gs.best_estimator_.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, proba)), gs.best_params_


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
    p.add_argument("--output-dir", default="results/h2/prompt_only_nonlinear")
    p.add_argument("--n-outer-splits", type=int, default=50)
    p.add_argument("--n-inner-folds", type=int, default=5)
    p.add_argument("--models", nargs="*",
                   default=["logistic", "random_forest", "gradient_boosting", "mlp"])
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[nonlinear-baseline] Loading samples from {args.hidden_states_dir} ...")
    samples = load_samples(Path(args.hidden_states_dir))
    y = np.array([int(s.passed) for s in samples], dtype=np.int32)
    pt = np.array([s.prompt_tokens for s in samples], dtype=np.float32).reshape(-1, 1)
    print(f"[nonlinear-baseline] {len(samples)} samples "
          f"(pass={int(y.sum())}, fail={int((1 - y).sum())}); "
          f"feature = prompt_tokens only")
    print(f"[nonlinear-baseline] models: {args.models}, "
          f"n_outer={args.n_outer_splits}, n_inner={args.n_inner_folds}")

    per_model_aucs: dict[str, list[float]] = {m: [] for m in args.models}

    for outer_seed in range(args.n_outer_splits):
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=outer_seed)
        X_tr, X_te = pt[tr], pt[te]
        y_tr, y_te = y[tr], y[te]
        for m in args.models:
            auc, _ = _fit_eval_one(m, X_tr, y_tr, X_te, y_te,
                                   cv_folds=args.n_inner_folds, seed=outer_seed)
            per_model_aucs[m].append(auc)
        if (outer_seed + 1) % 10 == 0:
            line = "  ".join(f"{m}={np.mean(per_model_aucs[m]):.3f}" for m in args.models)
            print(f"[nonlinear-baseline] outer {outer_seed+1}/{args.n_outer_splits}  {line}",
                  flush=True)

    print("\n" + "=" * 64)
    print(f"PROMPT-ONLY BASELINES (feature = prompt_tokens), n_outer={args.n_outer_splits}")
    print("=" * 64)
    print(f"{'Model':<20}{'Test AUC (mean ± 95% CI)':<28}{'[min, max]'}")
    print("-" * 64)
    summaries = {}
    for m in args.models:
        s = _summary(per_model_aucs[m])
        summaries[m] = s
        print(f"{m:<20}{s['mean']:.3f} ± {s['ci95_half']:.3f}{'':<14}[{s['min']:.3f}, {s['max']:.3f}]")
    print("=" * 64)
    print("Reference: linear logistic prompt-only baseline = 0.754; "
          "residualized probe = 0.911; raw probe = 0.931.")
    print("=" * 64)

    payload = {
        "n_outer_splits": args.n_outer_splits,
        "n_inner_folds": args.n_inner_folds,
        "feature": "prompt_tokens",
        "n_samples": len(samples),
        "models": args.models,
        "per_model_test_auc_summary": summaries,
        "per_model_test_auc_per_split": per_model_aucs,
        "reference": {"linear_baseline": 0.754, "residualized_probe": 0.911, "raw_probe": 0.931},
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[nonlinear-baseline] Saved {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
