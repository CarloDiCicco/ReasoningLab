"""
probing/data.py
───────────────
Load hidden-state .npz files produced by the experiment runner and build
feature matrices for linear probe training.

Each .npz file corresponds to one task and contains:
  - "attempt_<i>_layer_<L>" → np.ndarray of shape (hidden_dim,)
    (the hidden state of the LAST prompt token at layer L, attempt i)
  - "passed_<i>" → bool scalar (did the generated code pass the tests?)

For baseline runs (budget=1), there is exactly one attempt per task
so we only read attempt index 0.

Public API:
  ProbeSample          — one task's hidden states + label
  load_samples         — glob .npz dir → list[ProbeSample]
  build_feature_matrix — samples → (X, y) numpy arrays
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ProbeSample:
    """One task's hidden states and pass/fail label.

    layer_vectors maps each saved layer index to a 1D float32 array of
    shape (hidden_dim,) — 2560 for Qwen3-4B.  The runner saves layer 0
    (the embedding output) plus the last 8 transformer layers.
    """

    task_id: str
    source: str  # "HumanEval" | "LCB"
    layer_vectors: dict[int, np.ndarray]  # layer_idx → (hidden_dim,)
    passed: bool


# Regex to parse .npz keys like "attempt_0_layer_29".
# We only read attempt_0 because baseline policy runs once per task.
_LAYER_KEY_RE = re.compile(r"^attempt_0_layer_(\d+)$")


def _task_id_from_filename(filename: str) -> str:
    """Convert the sanitized .npz filename back to the original task_id.

    The runner replaces "/" with "_" before saving (filesystem safety).
    We restore it for known prefixes so task_ids match the dataset.

    Examples:
        "HumanEval_0.npz"   → "HumanEval/0"
        "LCB_3674.npz"      → "LCB/3674"
        "Unknown_foo.npz"   → "Unknown_foo"   (no slash restoration)
    """
    stem = Path(filename).stem  # drop the .npz extension
    for prefix in ("HumanEval", "LCB"):
        if stem.startswith(f"{prefix}_"):
            # Replace only the first underscore after the prefix with a slash.
            return f"{prefix}/{stem[len(prefix) + 1:]}"
    return stem


def _source_from_task_id(task_id: str) -> str:
    """Derive the benchmark source from the task_id prefix."""
    if task_id.startswith("HumanEval"):
        return "HumanEval"
    return "LCB"


def load_samples(hidden_states_dir: Path) -> list[ProbeSample]:
    """Load all .npz files from a directory into ProbeSample objects.

    Reads only attempt_0 (the single attempt from a baseline run).
    Files missing attempt_0 keys are silently skipped.

    Returns:
        List of ProbeSample sorted alphabetically by filename so the
        ordering is deterministic across runs.
    """
    samples: list[ProbeSample] = []

    # glob("*.npz") returns files in arbitrary order on most filesystems;
    # sorted() makes the result deterministic (alphabetical by filename).
    npz_files = sorted(hidden_states_dir.glob("*.npz"))
    if not npz_files:
        return samples

    for npz_path in npz_files:
        data = np.load(npz_path)  # lazy-loads, no full array copy yet

        # Collect all layer vectors for attempt 0.
        # Each key looks like "attempt_0_layer_29" → layer index 29.
        layer_vectors: dict[int, np.ndarray] = {}
        for key in data.files:
            m = _LAYER_KEY_RE.match(key)
            if m:
                layer_idx = int(m.group(1))
                layer_vectors[layer_idx] = data[key]  # shape: (hidden_dim,)

        if not layer_vectors:
            # File has no attempt_0 layer keys — skip (shouldn't happen in normal runs).
            continue

        # "passed_0" is a 0-dimensional bool array: bool(data["passed_0"]) → Python bool.
        passed_key = "passed_0"
        if passed_key not in data:
            continue

        task_id = _task_id_from_filename(npz_path.name)

        samples.append(
            ProbeSample(
                task_id=task_id,
                source=_source_from_task_id(task_id),
                layer_vectors=layer_vectors,
                passed=bool(data[passed_key]),
            )
        )

    return samples


def build_feature_matrix(
    samples: list[ProbeSample],
    layers: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) arrays from a list of ProbeSamples.

    For per-layer analysis, pass a single-element list (e.g. layers=[35]).
    For the concatenated probe, pass None to use all available layers —
    they are concatenated in ascending index order, so the output dim is
    n_layers * hidden_dim (e.g. 9 * 2560 = 23040).

    Args:
        samples: Non-empty list of ProbeSample objects.
        layers:  Which layer indices to include.  None = all layers in
                 the first sample, sorted ascending.

    Returns:
        X: float32 array of shape (n_samples, n_layers * hidden_dim).
        y: int32 array of shape (n_samples,), 0 = fail, 1 = pass.

    Raises:
        ValueError: if samples is empty or a requested layer is missing.
    """
    if not samples:
        raise ValueError("samples list is empty")

    # Default to all layers present in the first sample.
    if layers is None:
        layers = sorted(samples[0].layer_vectors.keys())

    rows: list[np.ndarray] = []
    labels: list[int] = []

    for sample in samples:
        # Collect one vector per requested layer, then concatenate into a
        # single 1D feature vector for this sample.
        vecs: list[np.ndarray] = []
        for layer_idx in layers:
            if layer_idx not in sample.layer_vectors:
                raise ValueError(
                    f"Layer {layer_idx} not found in sample {sample.task_id}. "
                    f"Available: {sorted(sample.layer_vectors.keys())}"
                )
            vecs.append(sample.layer_vectors[layer_idx])

        # np.concatenate([v1, v2, ...]) produces shape (n_layers * hidden_dim,).
        rows.append(np.concatenate(vecs))
        labels.append(int(sample.passed))  # True → 1, False → 0

    # Stack rows into a 2D matrix; cast to float32 for sklearn compatibility.
    X = np.stack(rows).astype(np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y