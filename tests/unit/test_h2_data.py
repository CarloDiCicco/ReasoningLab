"""Tests for probing/data.py — loading .npz files and building feature matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reasoninglab.probing.data import (
    ProbeSample,
    _source_from_task_id,
    _task_id_from_filename,
    build_averaged_feature_matrix,
    build_feature_matrix,
    load_samples,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

HIDDEN_DIM = 16  # small for tests
LAYERS = [0, 29, 30]


def _save_fake_npz(
    directory: Path,
    filename: str,
    passed: bool,
    hidden_dim: int = HIDDEN_DIM,
    layers: list[int] | None = None,
) -> Path:
    """Create a synthetic .npz matching the runner's key format."""
    if layers is None:
        layers = LAYERS
    arrays: dict[str, np.ndarray] = {}
    for layer_idx in layers:
        arrays[f"attempt_0_layer_{layer_idx}"] = np.random.randn(hidden_dim).astype(np.float32)
    arrays["passed_0"] = np.array(passed)
    path = directory / filename
    np.savez(path, **arrays)
    return path


# ── Tests: filename / task_id conversion ─────────────────────────────────────

def test_task_id_from_humaneval_filename():
    assert _task_id_from_filename("HumanEval_42.npz") == "HumanEval/42"


def test_task_id_from_lcb_filename():
    assert _task_id_from_filename("LCB_3674.npz") == "LCB/3674"


def test_task_id_from_unknown_prefix():
    # Unknown prefix: no slash restoration, just strip .npz.
    assert _task_id_from_filename("Other_123.npz") == "Other_123"


def test_source_from_task_id():
    assert _source_from_task_id("HumanEval/0") == "HumanEval"
    assert _source_from_task_id("LCB/3674") == "LCB"


# ── Tests: load_samples ─────────────────────────────────────────────────────

def test_load_samples_returns_correct_count(tmp_path: Path):
    _save_fake_npz(tmp_path, "HumanEval_0.npz", passed=True)
    _save_fake_npz(tmp_path, "LCB_3674.npz", passed=False)
    _save_fake_npz(tmp_path, "LCB_3700.npz", passed=True)

    samples = load_samples(tmp_path)
    assert len(samples) == 3


def test_load_samples_fields(tmp_path: Path):
    _save_fake_npz(tmp_path, "HumanEval_5.npz", passed=True)

    samples = load_samples(tmp_path)
    s = samples[0]
    assert s.task_id == "HumanEval/5"
    assert s.source == "HumanEval"
    assert s.passed is True
    assert sorted(s.layer_vectors.keys()) == LAYERS
    assert s.layer_vectors[0].shape == (HIDDEN_DIM,)


def test_load_samples_empty_dir(tmp_path: Path):
    assert load_samples(tmp_path) == []


def test_load_samples_sorted_by_task_id(tmp_path: Path):
    _save_fake_npz(tmp_path, "LCB_999.npz", passed=False)
    _save_fake_npz(tmp_path, "HumanEval_0.npz", passed=True)
    _save_fake_npz(tmp_path, "LCB_100.npz", passed=True)

    samples = load_samples(tmp_path)
    task_ids = [s.task_id for s in samples]
    # Sorted by filename (glob), which is alphabetical.
    assert task_ids == ["HumanEval/0", "LCB/100", "LCB/999"]


# ── Tests: build_feature_matrix ──────────────────────────────────────────────

def _make_samples(n: int = 5) -> list[ProbeSample]:
    """Create n synthetic ProbeSamples."""
    samples = []
    for i in range(n):
        vecs = {layer: np.random.randn(HIDDEN_DIM).astype(np.float32) for layer in LAYERS}
        samples.append(ProbeSample(
            task_id=f"test/{i}",
            source="test",
            layer_vectors=vecs,
            passed=i % 2 == 0,
        ))
    return samples


def test_build_feature_matrix_all_layers():
    samples = _make_samples(5)
    X, y = build_feature_matrix(samples)
    assert X.shape == (5, len(LAYERS) * HIDDEN_DIM)
    assert y.shape == (5,)
    assert X.dtype == np.float32
    assert y.dtype == np.int32


def test_build_feature_matrix_single_layer():
    samples = _make_samples(5)
    X, y = build_feature_matrix(samples, layers=[29])
    assert X.shape == (5, HIDDEN_DIM)


def test_build_feature_matrix_labels():
    samples = _make_samples(4)
    _, y = build_feature_matrix(samples)
    # i=0 pass, i=1 fail, i=2 pass, i=3 fail
    np.testing.assert_array_equal(y, [1, 0, 1, 0])


def test_build_feature_matrix_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        build_feature_matrix([])


def test_build_feature_matrix_missing_layer_raises():
    samples = _make_samples(2)
    with pytest.raises(ValueError, match="Layer 999"):
        build_feature_matrix(samples, layers=[999])


# ── Tests: build_averaged_feature_matrix ─────────────────────────────────────

def test_averaged_feature_matrix_shape():
    samples = _make_samples(5)
    X, y = build_averaged_feature_matrix(samples, layers=[0, 29])
    # Averaging two layers → output dim = HIDDEN_DIM (not 2 * HIDDEN_DIM).
    assert X.shape == (5, HIDDEN_DIM)
    assert y.shape == (5,)


def test_averaged_feature_matrix_values():
    """The output should be the element-wise mean of the selected layers."""
    samples = _make_samples(1)
    X, _ = build_averaged_feature_matrix(samples, layers=[0, 29])
    expected = np.mean(
        [samples[0].layer_vectors[0], samples[0].layer_vectors[29]], axis=0
    )
    np.testing.assert_allclose(X[0], expected, atol=1e-6)


def test_averaged_feature_matrix_labels():
    samples = _make_samples(4)
    _, y = build_averaged_feature_matrix(samples, layers=[29])
    np.testing.assert_array_equal(y, [1, 0, 1, 0])


def test_averaged_feature_matrix_empty_samples_raises():
    with pytest.raises(ValueError, match="samples.*empty"):
        build_averaged_feature_matrix([], layers=[29])


def test_averaged_feature_matrix_empty_layers_raises():
    samples = _make_samples(2)
    with pytest.raises(ValueError, match="layers.*empty"):
        build_averaged_feature_matrix(samples, layers=[])