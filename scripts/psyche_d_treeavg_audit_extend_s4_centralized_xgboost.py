#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. `research_datasets/` is
# gitignored; local research data, never committed to this repo.
"""Section 4 of .planning/debug/psyche-d-treeavg-audit-extend.md: a
centralized XGBoost baseline, essential because the existing centralized
reference uses LightGBM-DART while the federated tree experiments (TreeAvg,
FedXgbBagging) use XGBoost -- without this baseline, any gap between
centralized LightGBM and the federated XGBoost methods conflates (A) a
LightGBM-vs-XGBoost model-family effect with (B) a centralized-vs-federated
effect. This baseline isolates (A) so (B) and (C, TreeAvg-vs-FedXgbBagging)
can be interpreted cleanly.

Same fixed participant-level train/val/test split (loaded from the shared
split cache), same 152 features, same raw-feature/native-missing-value
handling (NO SimpleImputer -- matches step4/step5's XGBoost convention, not
step1's MLP/DART convention, since this baseline's whole purpose is to be
the XGBoost-family comparator), same XGB_BASE_PARAMS family as the federated
tree experiments (eta=0.1, max_depth=4, subsample=0.8, colsample_bytree=0.8,
tree_method=hist), same validation-AUROC-based model selection (never the
test set), num_boost_round budget of 150 to match the federated tree
ensembles' total tree count (5 clients x 30 rounds = 150) and the existing
LightGBM-DART baseline's n_estimators=150, for a fair capacity comparison.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import psyche_d_fedavg_vs_fedtree_common as C  # noqa: E402
from psyche_d_fedavg_vs_fedtree_step4_fed_xgb_bagging import (  # noqa: E402
    XGB_BASE_PARAMS,
    load_split_and_frame,
)
from psyche_d_fedavg_vs_fedtree_step5_tree_avg import extended_metrics  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
MAX_BOOST_ROUND = 150  # matches 5 clients x 30 rounds (federated tree ensembles) and LightGBM-DART n_estimators=150
EARLY_STOPPING_ROUNDS = 20  # same patience convention as step1's MLP (PATIENCE=20)


def run() -> dict:
    t0 = time.time()
    cache, labeled, feature_cols, y, train_idx, val_idx, test_idx = load_split_and_frame()

    train_df = labeled.loc[train_idx]
    val_df = labeled.loc[val_idx]
    test_df = labeled.loc[test_idx]
    y_train = y.loc[train_idx].to_numpy()
    y_val = y.loc[val_idx].to_numpy()
    y_test = y.loc[test_idx].to_numpy()

    X_train = train_df[feature_cols].to_numpy(dtype=np.float64)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float64)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float64)

    # Centralized class-imbalance handling: a single scale_pos_weight computed
    # directly from the full (centralized) train set -- no federated-safe
    # aggregation trick needed here since this IS the centralized baseline.
    total_pos = int((y_train == 1).sum())
    total_neg = int((y_train == 0).sum())
    scale_pos_weight = total_neg / max(total_pos, 1)
    params = dict(XGB_BASE_PARAMS, scale_pos_weight=scale_pos_weight)

    dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
    dval = xgb.DMatrix(X_val, label=y_val, missing=np.nan)
    dtest = xgb.DMatrix(X_test, missing=np.nan)

    evals_result = {}
    bst = xgb.train(
        params, dtrain, num_boost_round=MAX_BOOST_ROUND,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        evals_result=evals_result, verbose_eval=False,
    )
    best_iteration = bst.best_iteration  # 0-indexed; selected via validation AUC only

    val_pred = bst.predict(dval, iteration_range=(0, best_iteration + 1))
    test_pred = bst.predict(dtest, iteration_range=(0, best_iteration + 1))
    val_auroc = float(roc_auc_score(y_val, val_pred))

    centralized_xgb_test_metrics = extended_metrics("centralized_xgboost (test)", y_test, test_pred)
    centralized_xgb_val_metrics = extended_metrics("centralized_xgboost (val, for reference only)", y_val, val_pred)

    round_history = [
        {"round": i + 1, "val_auroc_running": None}  # placeholder; XGBoost's evals_result stores AUC, see below
        for i in range(len(evals_result["val"]["auc"]))
    ]
    for i, auc in enumerate(evals_result["val"]["auc"]):
        round_history[i]["val_auroc_running"] = float(auc)

    cited = C.ARM_B_CENTRALIZED_LGBM_DART_REFERENCE
    results = {
        "section": 4,
        "description": "Centralized XGBoost baseline (NEW) -- isolates LightGBM-vs-XGBoost model-family effects from centralized-vs-federated and TreeAvg-vs-FedXgbBagging effects.",
        "split_seed": cache["seed"],
        "n_features": len(feature_cols),
        "xgb_params": params,
        "max_boost_round": MAX_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "selected_boosting_round": int(best_iteration + 1),  # 1-indexed tree count, for readability alongside LightGBM-DART's n_estimators
        "best_val_auroc": val_auroc,
        "round_history_val_auc": round_history,
        "centralized_xgboost_test_metrics": centralized_xgb_test_metrics,
        "centralized_xgboost_val_metrics": centralized_xgb_val_metrics,
        "cited_lightgbm_dart_reference_5fold_pooled_oof": cited,
        "comparison": {
            "xgb_auroc_minus_cited_lightgbm_auroc": centralized_xgb_test_metrics["auroc"] - cited["auroc"],
        },
        # purely additive provenance, same pattern as step1/step4/step5 -- for
        # Section 10's participant-level bootstrap paired comparisons.
        "y_test": y_test.tolist(),
        "centralized_xgboost_test_predictions": test_pred.tolist(),
        "y_val": y_val.tolist(),
        "centralized_xgboost_val_predictions": val_pred.tolist(),
        "runtime_seconds": time.time() - t0,
    }

    out_path = SCRIPT_DIR / "psyche_d_treeavg_audit_extend_s4_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {out_path}")
    print(f"Centralized XGBoost: selected_round={results['selected_boosting_round']} val_auroc={val_auroc:.4f}")
    print(f"test AUROC={centralized_xgb_test_metrics['auroc']:.4f} AUPRC={centralized_xgb_test_metrics['auprc']:.4f} "
          f"bal_acc={centralized_xgb_test_metrics['balanced_accuracy']:.4f} brier={centralized_xgb_test_metrics['brier_score']:.4f}")
    print(f"vs cited LightGBM-DART (0.7090): delta={results['comparison']['xgb_auroc_minus_cited_lightgbm_auroc']:+.4f}")
    return results


if __name__ == "__main__":
    run()
