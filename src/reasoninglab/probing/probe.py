"""
probing/probe.py
────────────────
Train a linear probe (PCA + LogisticRegression) to predict pass/fail
from model hidden states.

Public API:
  ProbeResult  — frozen dataclass with metrics from one probe run
  train_probe  — fit scaler → PCA → logreg pipeline, evaluate, return results
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ProbeResult:
    """Metrics from one probe training run.

    Returned alongside a fitted sklearn Pipeline so the caller can both
    inspect numbers and save/reuse the trained model.
    """

    layer: int | str  # layer index (e.g. 35) or "concat" for all-layers run
    best_C: float     # regularization strength chosen by cross-validation (higher = less regularization)
    cv_auc_mean: float  # average AUC across CV folds for best_C (estimate of generalisation)
    cv_auc_std: float   # std across CV folds (lower = more stable estimate)
    test_auc: float     # AUC on the held-out test set (primary metric, threshold-independent)
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    n_train: int
    n_test: int
    n_features_raw: int    # feature dim BEFORE PCA (e.g. 2560 for one layer, 23040 for 9 layers)
    n_features_pca: int    # feature dim AFTER PCA (auto-selected to reach pca_variance threshold)
    pca_variance_explained: float  # actual cumulative variance retained (≈ pca_variance threshold)


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    pca_variance: float = 0.95,
    C_values: list[float] | None = None,
    cv_folds: int = 5,
    seed: int = 0,
    layer_label: int | str = "concat",
) -> tuple[ProbeResult, Pipeline]:
    """Train a StandardScaler → PCA → LogisticRegression pipeline.

    Pipeline overview:
      1. StandardScaler  — zero mean, unit variance per feature.
                           Critical because layer 0 (embedding) has range [-0.1, 0.1]
                           while layer 35 can be [-50, 250]. Without scaling, PCA would
                           be dominated by the high-magnitude late layers.
      2. PCA             — reduce dimensionality. 'pca_variance' is a float threshold
                           (e.g. 0.95), sklearn automatically selects the smallest k
                           such that the retained components explain ≥ 95% of variance.
                           This avoids manual guessing of k.
      3. LogisticRegression — L2-regularised binary classifier. C is the inverse of
                           regularization strength (C=0.01 = strong regularization,
                           C=10 = weak). GridSearchCV finds the best C.

    The returned Pipeline bundles all three steps so that it can be saved with
    joblib.dump() and later applied to new hidden states with a single
    pipeline.predict_proba(X_new) call — no need to reapply scaler/PCA manually.

    Args:
        X_train:      Training features, shape (n_train, n_features).
        y_train:      Training labels, shape (n_train,), values 0 (fail) or 1 (pass).
        X_test:       Test features, shape (n_test, n_features).
        y_test:       Test labels, shape (n_test,), values 0 or 1.
        pca_variance: Cumulative variance threshold (0.0–1.0). sklearn PCA
                      accepts a float < 1.0 as the number of components to keep.
        C_values:     Regularization strengths to grid-search over.
        cv_folds:     Number of stratified cross-validation folds.
        seed:         Random seed for CV shuffles and LogisticRegression.
        layer_label:  Stored in ProbeResult.layer — use an int for per-layer
                      runs and "concat" for the all-layers run.

    Returns:
        (ProbeResult, Pipeline) — metrics and the fitted sklearn pipeline.
        The Pipeline is only saved by the caller for the concat probe
        (the per-layer pipelines are discarded after reading their metrics).
    """
    if C_values is None:
        C_values = [0.001, 0.01, 0.1, 1.0, 10.0]

    n_features_raw = X_train.shape[1]

    # ── Step 1: Scale ──────────────────────────────────────────────────────────
    scaler = StandardScaler()
    # fit_transform on train: computes mean/std from training data and applies it.
    # transform on test: applies the SAME mean/std (never fit on test — data leakage).
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Step 2: PCA ────────────────────────────────────────────────────────────
    # Passing a float (0.0 < pca_variance < 1.0) tells sklearn to auto-select
    # the minimum number of components that explain ≥ pca_variance of the total
    # variance. The actual number of components is stored in pca.n_components_
    # after fitting.
    pca = PCA(n_components=pca_variance, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_scaled)  # fit on train, reduce both
    X_test_pca = pca.transform(X_test_scaled)

    n_features_pca = pca.n_components_
    # sum of explained_variance_ratio_ gives the actual cumulative variance kept.
    variance_explained = float(np.sum(pca.explained_variance_ratio_))

    # ── Step 3: Hyperparameter search over C ───────────────────────────────────
    # StratifiedKFold preserves the pass/fail ratio in every fold — important
    # because coding tasks have an imbalanced label distribution (~40% pass).
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed),
        param_grid={"C": C_values},
        cv=cv,
        scoring="roc_auc",  # AUC is threshold-independent, ideal for imbalanced labels
        refit=True,         # after search, refit the best model on the full training set
    )
    grid.fit(X_train_pca, y_train)

    best_C = grid.best_params_["C"]
    cv_auc_mean = float(grid.best_score_)
    # cv_results_ contains per-fold scores; we take the std for the best C index.
    best_idx = grid.best_index_
    cv_auc_std = float(grid.cv_results_["std_test_score"][best_idx])

    # ── Step 4: Evaluate on held-out test set ──────────────────────────────────
    # grid.best_estimator_ is the LogisticRegression already fitted on full train.
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_pca)
    # predict_proba returns [[p_class0, p_class1], ...]; we take column 1 (pass probability).
    y_proba = best_model.predict_proba(X_test_pca)[:, 1]

    test_auc = float(roc_auc_score(y_test, y_proba))
    test_accuracy = float(accuracy_score(y_test, y_pred))
    # zero_division=0.0 silences warnings when a class is never predicted.
    test_precision = float(precision_score(y_test, y_pred, zero_division=0.0))
    test_recall = float(recall_score(y_test, y_pred, zero_division=0.0))
    test_f1 = float(f1_score(y_test, y_pred, zero_division=0.0))

    # ── Build Pipeline for serialization ───────────────────────────────────────
    # Bundle all three fitted objects (scaler, pca, logreg) into one Pipeline.
    # This lets the caller save/load the entire probe with a single joblib call
    # and apply it to new hidden states as pipeline.predict_proba(X_new).
    # Note: we build the pipeline AFTER training rather than using sklearn's
    # Pipeline + GridSearchCV together, because we need to apply GridSearchCV
    # only to the LogisticRegression step (after PCA is already fitted).
    pipeline = Pipeline([
        ("scaler", scaler),    # already fitted on X_train
        ("pca", pca),          # already fitted on X_train_scaled
        ("logreg", best_model),  # already fitted on X_train_pca
    ])

    result = ProbeResult(
        layer=layer_label,
        best_C=best_C,
        cv_auc_mean=cv_auc_mean,
        cv_auc_std=cv_auc_std,
        test_auc=test_auc,
        test_accuracy=test_accuracy,
        test_precision=test_precision,
        test_recall=test_recall,
        test_f1=test_f1,
        n_train=X_train.shape[0],
        n_test=X_test.shape[0],
        n_features_raw=n_features_raw,
        n_features_pca=n_features_pca,
        pca_variance_explained=variance_explained,
    )

    return result, pipeline