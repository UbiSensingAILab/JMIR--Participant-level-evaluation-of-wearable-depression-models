#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. `research_datasets/` is
# gitignored; local research data, never committed to this repo.
"""Experiment B (task Part 6.B): bootstrap 95% CIs for AUROC/AUPRC on the
fixed held-out test set, plus PAIRED bootstrap comparisons between TreeAvg
and each of {centralized LightGBM, centralized MLP, FedAvg MLP,
FedXgbBagging} on the SAME test observations.

This script only CONSUMES the result JSONs already written by Steps 1/23/4
and the corrected Step 5 (sample-count-weighted n_k/N TreeAvg) -- there is
no TreeAvg algorithm logic in this file to get right or wrong; it inherits
the corrected weighting automatically via
`psyche_d_fedavg_vs_fedtree_step5_results.json`'s
`tree_avg_test_predictions`.

STATISTICAL PROCEDURE (stated explicitly, per task instruction):
  - Bootstrap: row-level resampling with replacement, n_boot=2000,
    numpy Generator seeded 42 (SAME seed/convention as this file family's
    existing `bootstrap_ci_auroc` in psyche_d_fedavg_vs_fedtree_common.py).
  - DISCLOSED LIMITATION: this is a ROW-level bootstrap, not a
    participant-level (cluster) bootstrap. The test set has 1631 rows from
    606 participants (up to 4 rows/participant, longitudinal). A row-level
    bootstrap treats each row as an independent resampling unit, which can
    UNDERSTATE the true sampling variability slightly, since a
    participant's multiple rows are correlated (similar wearable/context
    signal, identical baseline severity). A participant-level (cluster)
    bootstrap -- resampling whole participants with replacement, then
    taking all of that participant's rows -- would be the methodologically
    stricter choice. It was not implemented here in the time available for
    this task; row-level is used instead, MATCHING this project's own
    existing convention (`bootstrap_ci_auroc`, used by every prior
    psyche-d-fedavg-vs-fedtree script), so this report's new CIs are at
    least self-consistent with its own already-published ones. This choice
    is disclosed here, not silently made.
  - Paired comparison: for a given pair of methods (A, B) evaluated on the
    IDENTICAL 1631 test rows, each bootstrap iteration draws ONE shared
    resampled index set (with replacement) and computes AUROC_A - AUROC_B
    on that SAME resampled set for both methods (this is what makes it
    "paired" -- both methods see identical resampled data every iteration,
    canceling out shared sampling noise). The 95% CI of the resulting
    2000 deltas is reported; if the interval excludes 0, the difference is
    treated as statistically distinguishable at that discrete-level.
  - No claim of significance beyond what the CI actually shows -- deltas
    whose CI straddles zero are reported as "not distinguishable from
    zero", not softened into a qualitative verdict.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
N_BOOT = 2000
SEED = 42
REFERENCE_PARTITION_SEED = 1  # for the FL methods' paired comparison -- same seed used across FedAvg/FedXgbBagging/TreeAvg


def load_json(name: str) -> dict:
    return json.loads((SCRIPT_DIR / name).read_text())


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, metric_fn, n_boot=N_BOOT, seed=SEED) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y_true[idx], y_score[idx]
        if len(np.unique(yb)) < 2:
            continue
        vals.append(metric_fn(yb, sb))
    point = float(metric_fn(y_true, y_score))
    return {
        "point_estimate": point,
        "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
        "n_boot_used": len(vals),
    }


def paired_bootstrap_delta(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, metric_fn, n_boot=N_BOOT, seed=SEED) -> dict:
    """Both scores MUST be aligned to the identical row ordering of y_true."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        a = metric_fn(yb, score_a[idx])
        b = metric_fn(yb, score_b[idx])
        deltas.append(a - b)
    point_delta = float(metric_fn(y_true, score_a) - metric_fn(y_true, score_b))
    ci_lo, ci_hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    return {
        "point_delta": point_delta,
        "ci95": [ci_lo, ci_hi],
        "excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        "n_boot_used": len(deltas),
    }


def run() -> dict:
    step1 = load_json("psyche_d_fedavg_vs_fedtree_step1_results.json")
    step23 = load_json("psyche_d_fedavg_vs_fedtree_step23_results.json")
    step4 = load_json("psyche_d_fedavg_vs_fedtree_step4_results.json")
    step5 = load_json("psyche_d_fedavg_vs_fedtree_step5_results.json")

    y_test = np.array(step1["y_test"])  # identical across all steps -- same fixed test set, verified below
    assert step1["y_test"] == step23["per_seed_results"][0]["y_test"]
    assert step1["y_test"] == step4["per_seed_results"][0]["y_test"]
    assert step1["y_test"] == step5["per_seed_results"][0]["y_test"]

    centralized_lgbm = np.array(step1["lightgbm_dart_same_split_test_predictions"])
    centralized_mlp = np.array(step1["centralized_mlp_test_predictions"])

    def by_seed(step, key):
        return {r["partition_seed"]: np.array(r[key]) for r in step["per_seed_results"]}

    fedavg_by_seed = by_seed(step23, "fedavg_test_predictions")
    fedxgb_by_seed = by_seed(step4, "fed_xgb_bagging_test_predictions")
    treeavg_by_seed = by_seed(step5, "tree_avg_test_predictions")

    fedavg_ref = fedavg_by_seed[REFERENCE_PARTITION_SEED]
    fedxgb_ref = fedxgb_by_seed[REFERENCE_PARTITION_SEED]
    treeavg_ref = treeavg_by_seed[REFERENCE_PARTITION_SEED]

    auroc_ci = {
        "centralized_lightgbm_same_split": bootstrap_ci(y_test, centralized_lgbm, roc_auc_score),
        "centralized_mlp": bootstrap_ci(y_test, centralized_mlp, roc_auc_score),
        f"fedavg_mlp_seed{REFERENCE_PARTITION_SEED}": bootstrap_ci(y_test, fedavg_ref, roc_auc_score),
        f"fedxgb_bagging_seed{REFERENCE_PARTITION_SEED}": bootstrap_ci(y_test, fedxgb_ref, roc_auc_score),
        f"tree_avg_seed{REFERENCE_PARTITION_SEED}": bootstrap_ci(y_test, treeavg_ref, roc_auc_score),
    }
    auprc_ci = {
        "centralized_lightgbm_same_split": bootstrap_ci(y_test, centralized_lgbm, average_precision_score),
        "centralized_mlp": bootstrap_ci(y_test, centralized_mlp, average_precision_score),
        f"fedavg_mlp_seed{REFERENCE_PARTITION_SEED}": bootstrap_ci(y_test, fedavg_ref, average_precision_score),
        f"fedxgb_bagging_seed{REFERENCE_PARTITION_SEED}": bootstrap_ci(y_test, fedxgb_ref, average_precision_score),
        f"tree_avg_seed{REFERENCE_PARTITION_SEED}": bootstrap_ci(y_test, treeavg_ref, average_precision_score),
    }

    paired_vs_treeavg = {
        "treeavg_vs_centralized_lightgbm": {
            "auroc": paired_bootstrap_delta(y_test, treeavg_ref, centralized_lgbm, roc_auc_score),
            "auprc": paired_bootstrap_delta(y_test, treeavg_ref, centralized_lgbm, average_precision_score),
        },
        "treeavg_vs_centralized_mlp": {
            "auroc": paired_bootstrap_delta(y_test, treeavg_ref, centralized_mlp, roc_auc_score),
            "auprc": paired_bootstrap_delta(y_test, treeavg_ref, centralized_mlp, average_precision_score),
        },
        "treeavg_vs_fedavg_mlp": {
            "auroc": paired_bootstrap_delta(y_test, treeavg_ref, fedavg_ref, roc_auc_score),
            "auprc": paired_bootstrap_delta(y_test, treeavg_ref, fedavg_ref, average_precision_score),
        },
        "treeavg_vs_fedxgb_bagging": {
            "auroc": paired_bootstrap_delta(y_test, treeavg_ref, fedxgb_ref, roc_auc_score),
            "auprc": paired_bootstrap_delta(y_test, treeavg_ref, fedxgb_ref, average_precision_score),
        },
    }

    results = {
        "experiment": "B -- statistical uncertainty",
        "procedure": {
            "bootstrap_type": "row-level (NOT participant-level -- disclosed limitation, see module docstring)",
            "n_boot": N_BOOT, "seed": SEED,
            "reference_partition_seed_for_FL_methods": REFERENCE_PARTITION_SEED,
            "n_test_rows": int(len(y_test)), "n_test_participants": step1["split"]["n_test_participants"],
            "treeavg_weighting_scheme": "sample-count-weighted n_k/N (corrected) -- inherited from psyche_d_fedavg_vs_fedtree_step5_results.json, not recomputed here",
        },
        "auroc_bootstrap_ci95": auroc_ci,
        "auprc_bootstrap_ci95": auprc_ci,
        "paired_comparisons_vs_treeavg": paired_vs_treeavg,
    }

    out_path = SCRIPT_DIR / "psyche_d_fedavg_vs_fedtree_step5_experiment_b_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {out_path}")
    print("\n=== AUROC 95% CI ===")
    for k, v in auroc_ci.items():
        print(f"  {k}: {v['point_estimate']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]")
    print("\n=== Paired TreeAvg deltas (AUROC) ===")
    for k, v in paired_vs_treeavg.items():
        d = v["auroc"]
        print(f"  {k}: delta={d['point_delta']:+.4f} CI=[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] excludes_zero={d['excludes_zero']}")
    return results


if __name__ == "__main__":
    run()
