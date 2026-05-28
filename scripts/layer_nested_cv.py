#!/usr/bin/env python3
"""Nested CV with a joint (layer x C) grid search for the H2 probe.

Methodology (textbook nested CV per Cawley & Talbot 2010, JMLR; matches the
pattern used by sklearn's official 'Nested versus non-nested CV' tutorial):

  Outer loop (N stratified 80/20 splits, seeds 0..N-1):
    For each outer split:
      1. Set outer_test aside; it is touched ONLY at step 4.
      2. Inner loop on outer_train: 5-fold stratified CV. For each inner fold
         and each (layer, C) cell in the joint grid, fit
           StandardScaler -> PCA(0.95) -> LogReg(C=fixed)
         on inner-train, evaluate AUC on inner-val.
      3. Average across the 5 inner folds per cell. Pick (best_layer, best_C)
         = argmax over the grid means.
      4. Refit the same pipeline on the FULL outer-train at (best_layer, best_C)
         and evaluate ONCE on outer_test. Record outer-test AUC.

Reports (saved JSON + console table):
  - Headline: mean +/- 95% CI of outer-test AUC across N outer splits.
  - Layer-selection histogram (modal layer, per-layer frequencies).
  - C-selection histogram.
  - Joint (layer, C) selection histogram.
  - Per-layer mean inner-val AUC across all outer splits (diagnostic).

When --residualize is set:
  - Inside each inner fold, per layer, fit LinearRegression(pt -> X) on
    inner-train and subtract the prediction from both inner-train and inner-val.
    Re-fit on outer-train before the final refit at step 4.
  - Selection is on residualized inner-val AUC. The chosen (layer, C) reflects
    what is best FOR THE RESIDUALIZED CLAIM, which may differ from the raw run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from reasoninglab.probing.data import build_feature_matrix, load_samples

DEFAULT_C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


# ── Inner cell fitter ────────────────────────────────────────────────────────

def _fit_eval_cell(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    pca_variance: float,
    C: float,
    seed: int,
) -> float:
    """Fit StandardScaler -> PCA -> LogReg(C=fixed) on train, return val AUC.

    Matches the existing train_probe pipeline exactly, except that C is given
    (no internal GridSearchCV).
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_va = scaler.transform(X_val)

    pca = PCA(n_components=pca_variance, random_state=seed)
    X_tr = pca.fit_transform(X_tr)
    X_va = pca.transform(X_va)

    clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=seed)
    clf.fit(X_tr, y_train)
    proba = clf.predict_proba(X_va)[:, 1]
    return float(roc_auc_score(y_val, proba))


# ── Residualization helper ────────────────────────────────────────────────────

def _residualize(
    X_train: np.ndarray,
    X_val: np.ndarray,
    pt_train: np.ndarray,
    pt_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Regress prompt_tokens out of each X dimension, fit on train ONLY."""
    reg = LinearRegression()
    reg.fit(pt_train.reshape(-1, 1), X_train)
    X_train_r = (X_train - reg.predict(pt_train.reshape(-1, 1))).astype(np.float32)
    X_val_r = (X_val - reg.predict(pt_val.reshape(-1, 1))).astype(np.float32)
    return X_train_r, X_val_r


# ── Inner loop: joint (layer, C) grid selection ──────────────────────────────

def _inner_grid_selection(
    X_per_layer_train: dict[int, np.ndarray],
    pt_train: np.ndarray,
    y_train: np.ndarray,
    layers: list[int],
    c_grid: list[float],
    n_inner_folds: int,
    pca_variance: float,
    seed: int,
    residualize: bool,
) -> tuple[int, float, dict[tuple[int, float], float]]:
    """Return (best_layer, best_C, mean_inner_val_auc_per_cell).

    Cells are evaluated on the SAME inner partition for fair comparison
    (random_state = outer seed).
    """
    skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)
    # cell -> list of inner-fold AUCs
    cell_aucs: dict[tuple[int, float], list[float]] = {
        (L, C): [] for L in layers for C in c_grid
    }

    for inner_train_idx, inner_val_idx in skf.split(np.zeros(len(y_train)), y_train):
        y_it = y_train[inner_train_idx]
        y_iv = y_train[inner_val_idx]
        pt_it = pt_train[inner_train_idx] if residualize else None
        pt_iv = pt_train[inner_val_idx] if residualize else None

        for layer in layers:
            Xl = X_per_layer_train[layer]
            X_it = Xl[inner_train_idx]
            X_iv = Xl[inner_val_idx]
            if residualize:
                X_it, X_iv = _residualize(X_it, X_iv, pt_it, pt_iv)
            for C in c_grid:
                auc = _fit_eval_cell(
                    X_it, y_it, X_iv, y_iv,
                    pca_variance=pca_variance, C=C, seed=seed)
                cell_aucs[(layer, C)].append(auc)

    cell_means = {cell: float(np.mean(aucs)) for cell, aucs in cell_aucs.items()}
    best_cell = max(cell_means, key=cell_means.get)
    return best_cell[0], best_cell[1], cell_means


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Nested CV with joint (layer x C) grid for the H2 probe.")
    p.add_argument("hidden_states_dir")
    p.add_argument("--output-dir", default="results/h2/probe_lcb444_nested_cv_grid")
    p.add_argument("--n-outer-splits", type=int, default=50)
    p.add_argument("--n-inner-folds", type=int, default=5)
    p.add_argument("--pca-variance", type=float, default=0.95)
    p.add_argument("--c-grid", type=float, nargs="*", default=DEFAULT_C_GRID)
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--residualize", action="store_true",
                   help="Residualize prompt_tokens out of hidden states inside "
                        "each inner fold; selection becomes residualized-AUC-based.")
    return p.parse_args()


# ── Aggregation ───────────────────────────────────────────────────────────────

def _summary(arr: list[float]) -> dict:
    a = np.array(arr)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)),
        "ci95_half": float(1.96 * a.std(ddof=1) / np.sqrt(len(a))),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = "RESIDUALIZED" if args.residualize else "RAW"
    print(f"[nested-cv:{mode}] Loading samples from {args.hidden_states_dir} ...")
    samples = load_samples(Path(args.hidden_states_dir))
    available_layers = sorted(samples[0].layer_vectors.keys())
    layers = args.layers if args.layers else [l for l in available_layers if l != 0]
    c_grid = list(args.c_grid)
    print(f"[nested-cv:{mode}] {len(samples)} samples, candidate layers: {layers}")
    print(f"[nested-cv:{mode}] C grid: {c_grid}")
    print(f"[nested-cv:{mode}] n_outer={args.n_outer_splits}, n_inner={args.n_inner_folds}, "
          f"PCA={args.pca_variance}, cells_per_outer="
          f"{len(layers) * len(c_grid) * args.n_inner_folds}")

    # Pre-build per-layer feature matrices and prompt_tokens vector
    print(f"[nested-cv:{mode}] Building per-layer feature matrices ...")
    X_per_layer: dict[int, np.ndarray] = {}
    y_global: np.ndarray | None = None
    for layer in layers:
        X, y = build_feature_matrix(samples, [layer])
        X_per_layer[layer] = X
        if y_global is None:
            y_global = y
        else:
            assert np.array_equal(y, y_global), "Label mismatch across layers"
    y = y_global  # type: ignore[assignment]
    # prompt_tokens vector (one per sample, same order as samples)
    pt = np.array([s.prompt_tokens for s in samples], dtype=np.float32)
    assert len(pt) == len(y)

    # Outer loop
    outer_test_aucs: list[float] = []
    selected_layers: list[int] = []
    selected_Cs: list[float] = []
    per_outer_cell_means: list[dict[str, float]] = []  # key "L_C" for JSON-friendly

    for outer_seed in range(args.n_outer_splits):
        idx = np.arange(len(y))
        ot_idx, ote_idx = train_test_split(
            idx, test_size=0.2, stratify=y, random_state=outer_seed)
        y_ot, y_ote = y[ot_idx], y[ote_idx]
        pt_ot, pt_ote = pt[ot_idx], pt[ote_idx]
        X_ot_per_layer = {l: X_per_layer[l][ot_idx] for l in layers}
        X_ote_per_layer = {l: X_per_layer[l][ote_idx] for l in layers}

        best_layer, best_C, cell_means = _inner_grid_selection(
            X_ot_per_layer, pt_ot, y_ot, layers, c_grid,
            n_inner_folds=args.n_inner_folds,
            pca_variance=args.pca_variance,
            seed=outer_seed,
            residualize=args.residualize,
        )
        selected_layers.append(best_layer)
        selected_Cs.append(best_C)
        per_outer_cell_means.append(
            {f"{L}_{C}": cell_means[(L, C)] for L in layers for C in c_grid})

        # Final refit on full outer-train at (best_layer, best_C); evaluate on outer-test
        X_ot_L = X_ot_per_layer[best_layer]
        X_ote_L = X_ote_per_layer[best_layer]
        if args.residualize:
            X_ot_L, X_ote_L = _residualize(X_ot_L, X_ote_L, pt_ot, pt_ote)
        outer_auc = _fit_eval_cell(
            X_ot_L, y_ot, X_ote_L, y_ote,
            pca_variance=args.pca_variance, C=best_C, seed=outer_seed)
        outer_test_aucs.append(outer_auc)

        if (outer_seed + 1) % 5 == 0:
            print(f"[nested-cv:{mode}] outer {outer_seed+1}/{args.n_outer_splits}  "
                  f"selected=(L={best_layer}, C={best_C})  "
                  f"outer_test_auc={outer_auc:.3f}", flush=True)

    # Aggregate
    test_summary = _summary(outer_test_aucs)
    layer_counts = Counter(selected_layers)
    c_counts = Counter(selected_Cs)
    joint_counts = Counter(zip(selected_layers, selected_Cs))
    modal_layer, modal_layer_count = layer_counts.most_common(1)[0]
    modal_C, modal_C_count = c_counts.most_common(1)[0]
    modal_joint, modal_joint_count = joint_counts.most_common(1)[0]

    # Per-cell mean across all outer splits (diagnostic)
    per_cell_global_mean: dict[str, float] = {}
    for L in layers:
        for C in c_grid:
            key = f"{L}_{C}"
            vals = [d[key] for d in per_outer_cell_means]
            per_cell_global_mean[key] = float(np.mean(vals))
    # Per-layer marginal (average over C and over outer splits)
    per_layer_global_mean: dict[int, float] = {
        L: float(np.mean([per_cell_global_mean[f"{L}_{C}"] for C in c_grid]))
        for L in layers
    }
    # Per-C marginal
    per_C_global_mean: dict[float, float] = {
        C: float(np.mean([per_cell_global_mean[f"{L}_{C}"] for L in layers]))
        for C in c_grid
    }

    # Console report
    print("\n" + "=" * 80)
    print(f"NESTED-CV GRID RESULTS ({mode})  n_outer={args.n_outer_splits}")
    print("=" * 80)
    print(f"\nHeadline outer test AUC:  {test_summary['mean']:.3f} +/- "
          f"{test_summary['ci95_half']:.3f} (95% CI)")
    print(f"                          std={test_summary['std']:.3f}  "
          f"min={test_summary['min']:.3f}  max={test_summary['max']:.3f}")
    print(f"\nLayer-selection histogram (modal = layer {modal_layer}, "
          f"{modal_layer_count}/{args.n_outer_splits}):")
    for L in layers:
        cnt = layer_counts.get(L, 0)
        bar = "#" * cnt
        pct = 100 * cnt / args.n_outer_splits
        print(f"  layer {L}: {cnt:3d}/{args.n_outer_splits}  ({pct:5.1f}%)  {bar}")
    print(f"\nC-selection histogram (modal = C={modal_C}, "
          f"{modal_C_count}/{args.n_outer_splits}):")
    for C in c_grid:
        cnt = c_counts.get(C, 0)
        bar = "#" * cnt
        pct = 100 * cnt / args.n_outer_splits
        print(f"  C={C:>7g}: {cnt:3d}/{args.n_outer_splits}  ({pct:5.1f}%)  {bar}")
    print(f"\nModal joint selection: (L={modal_joint[0]}, C={modal_joint[1]}) "
          f"chosen {modal_joint_count}/{args.n_outer_splits} times")
    print(f"\nPer-layer mean inner-val AUC (marginal over C and outer splits):")
    for L in layers:
        marker = "  <-- modal" if L == modal_layer else ""
        print(f"  layer {L}: {per_layer_global_mean[L]:.4f}{marker}")
    print(f"\nPer-C mean inner-val AUC (marginal over layers and outer splits):")
    for C in c_grid:
        marker = "  <-- modal" if C == modal_C else ""
        print(f"  C={C:>7g}: {per_C_global_mean[C]:.4f}{marker}")
    print("=" * 80)

    payload = {
        "mode": mode,
        "n_outer_splits": args.n_outer_splits,
        "n_inner_folds": args.n_inner_folds,
        "pca_variance": args.pca_variance,
        "c_grid": c_grid,
        "candidate_layers": layers,
        "n_samples": len(samples),
        "outer_test_aucs": outer_test_aucs,
        "outer_test_auc_summary": test_summary,
        "selected_layers": selected_layers,
        "selected_Cs": selected_Cs,
        "layer_selection_counts": {str(L): layer_counts.get(L, 0) for L in layers},
        "C_selection_counts": {str(C): c_counts.get(C, 0) for C in c_grid},
        "joint_selection_counts": {
            f"{L}_{C}": joint_counts.get((L, C), 0) for L in layers for C in c_grid
        },
        "modal_layer": modal_layer,
        "modal_C": modal_C,
        "modal_joint": [modal_joint[0], modal_joint[1]],
        "per_cell_global_mean_inner_val_auc": per_cell_global_mean,
        "per_layer_global_mean_inner_val_auc": {str(L): per_layer_global_mean[L] for L in layers},
        "per_C_global_mean_inner_val_auc": {str(C): per_C_global_mean[C] for C in c_grid},
        "per_outer_cell_means": per_outer_cell_means,
    }
    out_path = out_dir / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[nested-cv:{mode}] Saved {out_path}")


if __name__ == "__main__":
    main()
