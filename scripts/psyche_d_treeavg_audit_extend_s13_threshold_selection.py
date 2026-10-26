#!/usr/bin/env python3
"""Section 13 of .planning/debug/psyche-d-treeavg-audit-extend.md: for every
threshold-dependent metric (balanced accuracy, sensitivity, specificity, F1),
select the decision threshold using VALIDATION data ONLY (maximizing
validation balanced accuracy over a fixed 0.005-0.995 grid), freeze it, then
evaluate ONCE on the test set. The test set is NEVER used to pick the
threshold. This corrects every prior script's hardcoded 0.5 threshold
(disclosed there as the "same threshold the shared helper itself already
uses", not previously selected via validation).

Exception (disclosed, not silent): "Local-only MLP" is 5 INDEPENDENT
per-client models (not a single deployable classifier) -- threshold
selection is comparatively less meaningful for an ensemble-of-independent-
models comparator, so it is reported at the conventional fixed 0.5 threshold
elsewhere in this report, not re-derived here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, roc_auc_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
THRESH_GRID = np.linspace(0.005, 0.995, 199)
REFERENCE_PARTITION_SEED = 1


def load(name: str) -> dict:
    return json.loads((SCRIPT_DIR / name).read_text())


def select_threshold(y_val: np.ndarray, val_score: np.ndarray) -> float:
    best_thresh, best_bal_acc = 0.5, -1.0
    for t in THRESH_GRID:
        pred = (val_score >= t).astype(int)
        bal_acc = balanced_accuracy_score(y_val, pred)
        if bal_acc > best_bal_acc:
            best_bal_acc, best_thresh = bal_acc, float(t)
    return best_thresh


def metrics_at_threshold(name: str, y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "name": name, "decision_threshold": threshold,
        "auroc": float(roc_auc_score(y_true, y_score)), "auprc": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)), "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": sens, "specificity": spec, "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def run() -> dict:
    step1 = load("psyche_d_fedavg_vs_fedtree_step1_results.json")
    step23 = load("psyche_d_fedavg_vs_fedtree_step23_results.json")
    step4 = load("psyche_d_fedavg_vs_fedtree_step4_results.json")
    step5 = load("psyche_d_fedavg_vs_fedtree_step5_results.json")
    s4_xgb = load("psyche_d_treeavg_audit_extend_s4_results.json")
    s5_grid = load("psyche_d_treeavg_audit_extend_s5_results.json")

    def by_seed(step, key):
        return {r["partition_seed"]: np.array(r[key]) for r in step["per_seed_results"]}

    y_val_mlp = np.array(step1["y_val"])
    y_test_mlp = np.array(step1["y_test"])
    y_val_fed = by_seed(step4, "y_val")[REFERENCE_PARTITION_SEED]  # identical across steps, kept per-step for provenance
    y_test_fed = by_seed(step4, "y_test")[REFERENCE_PARTITION_SEED]

    models = {
        "centralized_lightgbm_same_split": {
            "y_val": y_val_mlp, "val_score": np.array(step1["lightgbm_dart_same_split_val_predictions"]),
            "y_test": y_test_mlp, "test_score": np.array(step1["lightgbm_dart_same_split_test_predictions"]),
        },
        "centralized_mlp": {
            "y_val": y_val_mlp, "val_score": np.array(step1["centralized_mlp_val_predictions"]),
            "y_test": y_test_mlp, "test_score": np.array(step1["centralized_mlp_test_predictions"]),
        },
        "centralized_xgboost": {
            "y_val": np.array(s4_xgb["y_val"]), "val_score": np.array(s4_xgb["centralized_xgboost_val_predictions"]),
            "y_test": np.array(s4_xgb["y_test"]), "test_score": np.array(s4_xgb["centralized_xgboost_test_predictions"]),
        },
        "fedavg_mlp": {
            "y_val": by_seed(step23, "y_val")[REFERENCE_PARTITION_SEED], "val_score": by_seed(step23, "fedavg_val_predictions")[REFERENCE_PARTITION_SEED],
            "y_test": by_seed(step23, "y_test")[REFERENCE_PARTITION_SEED], "test_score": by_seed(step23, "fedavg_test_predictions")[REFERENCE_PARTITION_SEED],
        },
        "fedxgb_bagging": {
            "y_val": y_val_fed, "val_score": by_seed(step4, "fed_xgb_bagging_val_predictions")[REFERENCE_PARTITION_SEED],
            "y_test": y_test_fed, "test_score": by_seed(step4, "fed_xgb_bagging_test_predictions")[REFERENCE_PARTITION_SEED],
        },
        "tree_avg_original_30rd_eta0.10": {
            "y_val": y_val_fed, "val_score": by_seed(step5, "tree_avg_val_predictions")[REFERENCE_PARTITION_SEED],
            "y_test": y_test_fed, "test_score": by_seed(step5, "tree_avg_test_predictions")[REFERENCE_PARTITION_SEED],
        },
        "tree_avg_selected_config": {
            "y_val": y_val_fed, "val_score": np.array(s5_grid["grid_results_by_eta_then_seed"][str(s5_grid["selected_eta"])][str(REFERENCE_PARTITION_SEED)]["val_predictions"]),
            "y_test": y_test_fed, "test_score": np.array(s5_grid["grid_results_by_eta_then_seed"][str(s5_grid["selected_eta"])][str(REFERENCE_PARTITION_SEED)]["test_predictions"]),
        },
    }

    results = {
        "section": 13,
        "description": "Validation-selected (balanced-accuracy-maximizing) decision threshold per model, frozen before touching test. Local-only MLP intentionally excluded (disclosed exception, see module docstring) -- reported at the conventional 0.5 threshold elsewhere.",
        "threshold_grid": "linspace(0.005, 0.995, 199)",
        "selection_criterion": "argmax balanced_accuracy on VALIDATION predictions only",
        "models": {},
    }
    for name, d in models.items():
        thresh = select_threshold(d["y_val"], d["val_score"])
        test_metrics = metrics_at_threshold(name, d["y_test"], d["test_score"], thresh)
        val_metrics = metrics_at_threshold(f"{name}_val_reference", d["y_val"], d["val_score"], thresh)
        results["models"][name] = {"selected_threshold": thresh, "test_metrics": test_metrics, "val_metrics_at_selected_threshold": val_metrics}

    out_path = SCRIPT_DIR / "psyche_d_treeavg_audit_extend_s13_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {out_path}")
    for name, d in results["models"].items():
        tm = d["test_metrics"]
        print(f"{name:38s} thresh={d['selected_threshold']:.3f} AUROC={tm['auroc']:.4f} bal_acc={tm['balanced_accuracy']:.4f} sens={tm['sensitivity']:.4f} spec={tm['specificity']:.4f} f1={tm['f1']:.4f} brier={tm['brier_score']:.4f}")
    return results


if __name__ == "__main__":
    run()
