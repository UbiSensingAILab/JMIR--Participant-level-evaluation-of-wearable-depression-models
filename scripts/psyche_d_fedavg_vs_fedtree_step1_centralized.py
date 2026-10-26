#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. `research_datasets/` is
# gitignored; local research data, never committed to this repo.
"""Step 1 of `.planning/debug/psyche-d-fedavg-vs-fedtree.md`: train a small,
FedAvg-compatible MLP centrally on the arm-B 152-feature matrix, using the
SAME ONE-TIME participant-level train/validation/test split that Steps 2-4
will reuse (built once here, cached to disk, never rebuilt/re-rolled).

Also fits the frozen arm-B LightGBM-DART recipe (`REPRO.phase2_dart_factory`)
on the EXACT SAME split as a same-split secondary check -- this is NOT the
cited StratifiedGroupKFold(5) pooled-OOF reference (0.7090); it is an
additional, honestly-labeled apples-to-apples number on this session's own
held-out test set, since the cited reference uses a different (rotating,
whole-population) evaluation protocol.

No broad hyperparameter search (per task instruction): ONE fixed MLP
architecture (see psyche_d_fedavg_vs_fedtree_mlp_model.py), early-stopped on
the validation set (never the test set) by AUROC, patience=20, max_epochs=300.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_psyche_d_baseline_reproduction as REPRO  # noqa: E402
import psyche_d_fedavg_vs_fedtree_common as C  # noqa: E402
import psyche_d_fedavg_vs_fedtree_mlp_model as M  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
SPLIT_CACHE_PATH = SCRIPT_DIR / "psyche_d_fedavg_vs_fedtree_split_cache.pkl"
MAX_EPOCHS = 300
PATIENCE = 20
LOCAL_EPOCHS_PER_CHECK = 1
LR = 1e-3


def build_and_cache_split() -> dict:
    """Builds the fixed split ONCE and caches participant sets + row indices
    to disk so every subsequent step (2/3/4) loads the IDENTICAL split
    instead of re-deriving it (protects against any accidental drift)."""
    labeled, feature_cols, y = C.load_arm_b_frame()
    split = C.build_fixed_split(labeled)
    cache = {
        "feature_cols": feature_cols,
        "train_participants": split["train_participants"],
        "val_participants": split["val_participants"],
        "test_participants": split["test_participants"],
        "seed": split["seed"],
    }
    SPLIT_CACHE_PATH.write_bytes(pickle.dumps(cache))
    print(f"Wrote {SPLIT_CACHE_PATH}")
    return split, labeled, feature_cols, y


def run() -> dict:
    t0 = time.time()
    split, labeled, feature_cols, y = build_and_cache_split()

    train_df = labeled.loc[split["train_idx"]]
    val_df = labeled.loc[split["val_idx"]]
    test_df = labeled.loc[split["test_idx"]]
    y_train = y.loc[split["train_idx"]].to_numpy()
    y_val = y.loc[split["val_idx"]].to_numpy()
    y_test = y.loc[split["test_idx"]].to_numpy()

    # Federated-safe preprocessing: sufficient stats computed from the TRAIN
    # pool only (val/test never touch imputation/scaling fitting). Called via
    # the SAME function Steps 2/3 use with K groups -- here with a single
    # group (centralized), proven identical to K-group aggregation in the
    # common-module smoke test.
    stats = C.federated_sufficient_stats([train_df], feature_cols)
    mean, std = stats["mean"], stats["std"]

    X_train = C.apply_impute_scale(train_df, feature_cols, mean, std)
    X_val = C.apply_impute_scale(val_df, feature_cols, mean, std)
    X_test = C.apply_impute_scale(test_df, feature_cols, mean, std)

    # --- Centralized MLP, early-stopped on val AUROC ---
    model = M.new_model(seed=M.SEED)
    best_val_auroc = -1.0
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
    history = []
    for epoch in range(MAX_EPOCHS):
        M.train_local(model, X_train, y_train, epochs=LOCAL_EPOCHS_PER_CHECK, lr=LR, seed=epoch)
        val_proba = M.predict_proba(model, X_val)
        val_auroc = float(roc_auc_score(y_val, val_proba))
        history.append({"epoch": epoch, "val_auroc": val_auroc})
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = M.get_state(model)
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, best val AUROC {best_val_auroc:.4f})")
                break
    M.set_state(model, best_state)
    test_proba_mlp = M.predict_proba(model, X_test)
    mlp_metrics = C.compute_metrics("centralized_mlp (test)", y_test, test_proba_mlp)
    val_proba_final = M.predict_proba(model, X_val)
    mlp_val_metrics = C.compute_metrics("centralized_mlp (val, for reference only)", y_val, val_proba_final)

    # --- Same-split LightGBM-DART secondary check (apples-to-apples on THIS split) ---
    from sklearn.impute import SimpleImputer
    from sklearn.utils.class_weight import compute_sample_weight
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    X_train_raw_imp = imputer.fit_transform(train_df[feature_cols])
    X_val_raw_imp = imputer.transform(val_df[feature_cols])
    X_test_raw_imp = imputer.transform(test_df[feature_cols])
    sw = compute_sample_weight(class_weight="balanced", y=y_train)
    clf_dart = REPRO.phase2_dart_factory()
    clf_dart.fit(X_train_raw_imp, y_train, sample_weight=sw)
    val_proba_dart = clf_dart.predict_proba(X_val_raw_imp)[:, 1]
    test_proba_dart = clf_dart.predict_proba(X_test_raw_imp)[:, 1]
    dart_same_split_metrics = C.compute_metrics("lightgbm_dart_SAME_SPLIT_secondary_check (test)", y_test, test_proba_dart)

    cited = C.ARM_B_CENTRALIZED_LGBM_DART_REFERENCE
    delta_vs_cited = mlp_metrics["auroc"] - cited["auroc"]
    delta_vs_same_split_dart = mlp_metrics["auroc"] - dart_same_split_metrics["auroc"]
    competitive_cited = abs(delta_vs_cited) <= 0.03
    competitive_same_split = abs(delta_vs_same_split_dart) <= 0.03

    results = {
        "step": 1,
        "description": "Centralized MLP on arm-B 152-feature matrix, fixed one-time participant train/val/test split (NOT K-fold). Same-split LightGBM-DART secondary check included for apples-to-apples comparison.",
        "architecture": {"input_dim": M.INPUT_DIM, "hidden": [M.HIDDEN1, M.HIDDEN2], "dropout": M.DROPOUT, "n_params": M.count_params(M.new_model())},
        "split": {
            "seed": split["seed"], "train_frac": C.TRAIN_FRAC, "val_frac": C.VAL_FRAC, "test_frac": C.TEST_FRAC,
            "n_train_participants": split["n_train_participants"], "n_val_participants": split["n_val_participants"], "n_test_participants": split["n_test_participants"],
            "n_train_rows": split["n_train_rows"], "n_val_rows": split["n_val_rows"], "n_test_rows": split["n_test_rows"],
        },
        "preprocessing": "Mean-impute + standardize, stats fit via federated_sufficient_stats([train_df]) -- train-pool-only, val/test never touched during fitting.",
        "training": {"max_epochs": MAX_EPOCHS, "patience": PATIENCE, "lr": LR, "best_epoch": best_epoch, "best_val_auroc": best_val_auroc, "n_epochs_run": len(history)},
        "centralized_mlp_test_metrics": mlp_metrics,
        "centralized_mlp_val_metrics": mlp_val_metrics,
        "lightgbm_dart_same_split_secondary_check_test_metrics": dart_same_split_metrics,
        "cited_lightgbm_dart_reference_5fold_pooled_oof": cited,
        # Added for Step 5's Experiment B (bootstrap CIs + paired comparisons) --
        # purely additive provenance, no existing metric/control-flow touched.
        "y_test": y_test.tolist(),
        "centralized_mlp_test_predictions": test_proba_mlp.tolist(),
        "lightgbm_dart_same_split_test_predictions": test_proba_dart.tolist(),
        # Added for the psyche-d-treeavg-audit-extend session's Section 13
        # (validation-selected decision threshold, never tuned on test) --
        # again purely additive provenance, no existing control-flow touched.
        "y_val": y_val.tolist(),
        "centralized_mlp_val_predictions": val_proba_final.tolist(),
        "lightgbm_dart_same_split_val_predictions": val_proba_dart.tolist(),
        "comparison": {
            "mlp_auroc_minus_cited_lightgbm_auroc": delta_vs_cited,
            "mlp_auroc_minus_same_split_lightgbm_auroc": delta_vs_same_split_dart,
            "competitive_screen_tolerance": 0.03,
            "competitive_vs_cited_reference": competitive_cited,
            "competitive_vs_same_split_lightgbm": competitive_same_split,
            "note": "Screening criterion only (task's own 0.02-0.03 tolerance) -- continuous difference reported regardless of pass/fail, and Steps 2/3/4 are run REGARDLESS of this screen's outcome per the task's Final Report requirements.",
        },
        "runtime_seconds": time.time() - t0,
    }

    json_path = SCRIPT_DIR / "psyche_d_fedavg_vs_fedtree_step1_results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {json_path}")
    print(f"\nMLP test AUROC={mlp_metrics['auroc']:.4f}  cited LightGBM AUROC={cited['auroc']:.4f}  same-split LightGBM AUROC={dart_same_split_metrics['auroc']:.4f}")
    print(f"delta vs cited: {delta_vs_cited:+.4f}  delta vs same-split LightGBM: {delta_vs_same_split_dart:+.4f}")
    print(f"competitive (within +/-0.03)? vs cited: {competitive_cited}  vs same-split: {competitive_same_split}")
    return results


if __name__ == "__main__":
    run()
