"""Tests for probing/probe.py — PCA + LogisticRegression training pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from reasoninglab.probing.probe import ProbeResult, train_probe


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_data(
    n: int = 100,
    dim: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic train/test data with ~50% positive rate."""
    rng = np.random.RandomState(seed)

    # Create two clusters with slight separation so AUC > 0.5 is achievable.
    X = rng.randn(n, dim).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.randn(n) > 0).astype(np.int32)

    split = int(0.8 * n)
    return X[:split], y[:split], X[split:], y[split:]


# ── Tests ────────────────────────────────────────────────────────────────────

def test_train_probe_returns_valid_result():
    X_train, y_train, X_test, y_test = _make_data()
    result, pipeline = train_probe(
        X_train, y_train, X_test, y_test,
        pca_variance=0.95,
        cv_folds=3,
        seed=0,
        layer_label=0,
    )

    assert isinstance(result, ProbeResult)
    assert result.layer == 0
    assert 0.0 <= result.test_auc <= 1.0
    assert 0.0 <= result.test_accuracy <= 1.0
    assert 0.0 <= result.cv_auc_mean <= 1.0
    assert result.cv_auc_std >= 0.0
    assert result.n_train == X_train.shape[0]
    assert result.n_test == X_test.shape[0]
    assert result.n_features_raw == X_train.shape[1]
    assert result.n_features_pca > 0
    assert result.pca_variance_explained >= 0.0


def test_train_probe_auc_above_chance():
    """With a linearly separable signal, AUC should beat random (0.5)."""
    X_train, y_train, X_test, y_test = _make_data(n=200, dim=10, seed=0)
    result, _ = train_probe(
        X_train, y_train, X_test, y_test,
        pca_variance=0.99,
        cv_folds=3,
        seed=0,
    )
    # With the planted signal in dim 0, AUC should be well above 0.5.
    assert result.test_auc > 0.6


def test_train_probe_pca_variance_respected():
    X_train, y_train, X_test, y_test = _make_data(n=100, dim=50)
    result, _ = train_probe(
        X_train, y_train, X_test, y_test,
        pca_variance=0.90,
        cv_folds=3,
        seed=0,
    )
    # PCA should keep fewer components than the full dimensionality.
    assert result.n_features_pca < 50
    assert result.pca_variance_explained >= 0.89  # allow small float tolerance


def test_train_probe_pca_caps_at_n_samples():
    """When n_samples < n_features, PCA components should be capped."""
    # 20 samples, 100 features → PCA can't have more than 20 components.
    rng = np.random.RandomState(0)
    X = rng.randn(20, 100).astype(np.float32)
    y = rng.randint(0, 2, size=20).astype(np.int32)
    # Ensure at least 2 classes in both splits.
    y[:10] = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y[10:] = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    result, _ = train_probe(
        X[:15], y[:15], X[15:], y[15:],
        pca_variance=0.99,
        cv_folds=2,
        seed=0,
    )
    assert result.n_features_pca <= 15


def test_train_probe_pipeline_serializable(tmp_path: Path):
    X_train, y_train, X_test, y_test = _make_data(n=60, dim=10)
    _, pipeline = train_probe(
        X_train, y_train, X_test, y_test,
        pca_variance=0.95,
        cv_folds=3,
        seed=0,
    )

    # Round-trip through joblib.
    path = tmp_path / "probe.joblib"
    joblib.dump(pipeline, path)
    loaded = joblib.load(path)

    # Loaded pipeline should produce identical predictions.
    proba_original = pipeline.predict_proba(X_test)
    proba_loaded = loaded.predict_proba(X_test)
    np.testing.assert_array_almost_equal(proba_original, proba_loaded)


def test_train_probe_concat_label():
    X_train, y_train, X_test, y_test = _make_data(n=60, dim=10)
    result, _ = train_probe(
        X_train, y_train, X_test, y_test,
        cv_folds=3,
        seed=0,
        layer_label="concat",
    )
    assert result.layer == "concat"
