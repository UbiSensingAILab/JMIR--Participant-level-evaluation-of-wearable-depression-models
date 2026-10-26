#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. `research_datasets/` is
# gitignored; local research data, never committed to this repo.
"""Experiment F (task Part 6.F): a smaller CONFIRMATORY grouped-CV run for
TreeAvg, to check whether the fixed-split conclusion (Experiment A/B) is
stable across participant-level folds -- NOT a replacement for the fixed
split, a separate confirmatory check per task instruction.

This file has its OWN inline reimplementation of TreeAvg's per-round
combination loop (it does not import `run_one_seed` from
`psyche_d_fedavg_vs_fedtree_step5_tree_avg`, since the fold/inner-holdout
structure here differs from the fixed-split protocol) -- so the
sample-count-weighted (n_k/N) combination rule is applied EXPLICITLY here
too, recomputed per fold (client row counts differ fold-to-fold, exactly
as they differ seed-to-seed in the fixed-split experiment).

SCOPE REDUCTION (disclosed, per task's own allowance -- "use fewer seeds
if necessary, but explain exactly what was done"):
  - StratifiedGroupKFold(n_splits=3) (not 5 -- the cited centralized
    LightGBM-DART reference used 5-fold; 3-fold is used here purely to
    bound TreeAvg's grouped-CV compute, and is explicitly NOT presented as
    the same protocol as the cited 0.7090 reference).
  - ONE client-partition seed per fold (seed=1), not 3 -- partition-seed
    variance was already characterized on the fixed split in Experiment A;
    this experiment's purpose is fold-stability, not re-establishing
    partition variance.
  - Within each outer fold's TRAIN portion, an inner 85/15 participant-level
    holdout is carved out purely for TreeAvg's best-round (early-stopping)
    selection -- the outer fold's TEST portion is NEVER used for anything
    but final scoring, exactly like the fixed-split protocol's val/test
    separation.
  - The fixed test set from Experiments A-E is NEVER reused here -- this
    experiment uses ALL participants (the full ~10,866-row/4,036-
    participant arm-B population), rotated through 3 outer folds, i.e. a
    disjoint, independent evaluation scheme.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import psyche_d_fedavg_vs_fedtree_common as C  # noqa: E402
from psyche_d_fedavg_vs_fedtree_step4_fed_xgb_bagging import (  # noqa: E402
    NUM_LOCAL_ROUND,
    XGB_BASE_PARAMS,
)
from psyche_d_fedavg_vs_fedtree_step5_tree_avg import extended_metrics, sigmoid  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
N_SPLITS = 3
CLIENT_PARTITION_SEED = 1
N_ROUNDS_F = 30  # same as Experiment A's TreeAvg (30 rounds)
INNER_VAL_FRAC = 0.15


def run_treeavg_fold(train_participants: set, test_idx: np.ndarray, labeled, feature_cols, y) -> dict:
    train_participants = np.array(sorted(train_participants))
    # inner participant-level holdout for best-round selection (never the outer test set)
    inner_strata = labeled[labeled["user_id"].isin(train_participants)].groupby("user_id")["phq9_cat_start"].agg(lambda s: s.mode().iloc[0])
    inner_train_p, inner_val_p = train_test_split(
        inner_strata.index.to_numpy(), train_size=1 - INNER_VAL_FRAC, random_state=20260928, stratify=inner_strata.to_numpy(),
    )
    inner_train_p, inner_val_p = set(inner_train_p), set(inner_val_p)

    sites = C.assign_clients(inner_train_p, k=C.N_CLIENTS, seed=CLIENT_PARTITION_SEED)
    train_df = labeled.loc[labeled["user_id"].isin(inner_train_p)]
    val_df = labeled.loc[labeled["user_id"].isin(inner_val_p)]
    test_df = labeled.loc[test_idx]
    y_val = y.loc[val_df.index].to_numpy()
    y_test = y.loc[test_idx].to_numpy()

    client_dfs = [train_df[train_df["user_id"].isin(s)] for s in sites]
    total_pos = sum(int((y.loc[d.index] == 1).sum()) for d in client_dfs)
    total_neg = sum(int((y.loc[d.index] == 0).sum()) for d in client_dfs)
    scale_pos_weight = total_neg / max(total_pos, 1)
    params = dict(XGB_BASE_PARAMS, scale_pos_weight=scale_pos_weight)
    K = C.N_CLIENTS

    # Sample-count-weighted (n_k/N) TreeAvg weights, RECOMPUTED for THIS
    # fold (client row counts differ fold-to-fold, since inner_train_p is a
    # different participant set each outer fold) -- mirrors the fixed-split
    # script's own per-seed recomputation.
    client_n_rows = np.array([len(d) for d in client_dfs], dtype=np.float64)
    N_total_rows = float(client_n_rows.sum())
    client_weights = client_n_rows / N_total_rows

    client_dmatrices, client_margin = [], []
    for d in client_dfs:
        yc = y.loc[d.index].to_numpy()
        Xc = d[feature_cols].to_numpy(dtype=np.float64)
        dm = xgb.DMatrix(Xc, label=yc, missing=np.nan)
        dm.set_base_margin(np.zeros(len(yc)))
        client_dmatrices.append(dm)
        client_margin.append(np.zeros(len(yc)))
    client_X = [d[feature_cols].to_numpy(dtype=np.float64) for d in client_dfs]
    d_client_eval = [xgb.DMatrix(Xc, missing=np.nan) for Xc in client_X]

    X_val = val_df[feature_cols].to_numpy(dtype=np.float64)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float64)
    d_val_eval = xgb.DMatrix(X_val, missing=np.nan)
    d_test_eval = xgb.DMatrix(X_test, missing=np.nan)

    val_margin = np.zeros(len(y_val))
    test_margin = np.zeros(len(y_test))
    best_val_auroc, best_round = -1.0, -1
    best_test_margin_snapshot = None

    for rnd in range(1, N_ROUNDS_F + 1):
        round_boosters = []
        for k in range(K):
            client_dmatrices[k].set_base_margin(client_margin[k])
            bst = xgb.train(params, client_dmatrices[k], num_boost_round=NUM_LOCAL_ROUND)
            round_boosters.append(bst)
        round_contrib_train = [
            np.sum(
                [client_weights[k2] * bst.predict(d_client_eval[k], output_margin=True) for k2, bst in enumerate(round_boosters)],
                axis=0,
            )
            for k in range(K)
        ]
        round_contrib_val = np.sum(
            [client_weights[k2] * bst.predict(d_val_eval, output_margin=True) for k2, bst in enumerate(round_boosters)],
            axis=0,
        )
        round_contrib_test = np.sum(
            [client_weights[k2] * bst.predict(d_test_eval, output_margin=True) for k2, bst in enumerate(round_boosters)],
            axis=0,
        )
        for k in range(K):
            client_margin[k] = client_margin[k] + round_contrib_train[k]
        val_margin = val_margin + round_contrib_val
        test_margin = test_margin + round_contrib_test
        val_auroc = float(roc_auc_score(y_val, sigmoid(val_margin)))
        if val_auroc > best_val_auroc:
            best_val_auroc, best_round = val_auroc, rnd
            best_test_margin_snapshot = test_margin.copy()

    test_pred = sigmoid(best_test_margin_snapshot)
    return {"y_test": y_test, "test_pred": test_pred, "best_round": best_round, "best_val_auroc": best_val_auroc,
            "client_n_participants": [len(s) for s in sites], "client_n_rows": [int(n) for n in client_n_rows.tolist()],
            "client_weights_n_k_over_N": client_weights.tolist(), "n_test_rows": len(y_test)}


def run() -> dict:
    t0 = time.time()
    labeled, feature_cols, y = C.load_arm_b_frame()
    participant_severity = labeled.groupby("user_id")["phq9_cat_start"].agg(lambda s: s.mode().iloc[0])
    participants = participant_severity.index.to_numpy()
    row_participant = labeled["user_id"].to_numpy()
    row_strata = labeled["phq9_cat_start"].to_numpy()  # stratify rows by baseline severity, group by participant

    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=20260928)
    fold_results = []
    all_y, all_pred = [], []
    for fold_i, (train_row_pos, test_row_pos) in enumerate(skf.split(np.zeros(len(labeled)), row_strata, groups=row_participant)):
        train_idx_labels = labeled.index[train_row_pos]
        test_idx_labels = labeled.index[test_row_pos]
        train_participants_fold = set(labeled.loc[train_idx_labels, "user_id"].unique())
        test_participants_fold = set(labeled.loc[test_idx_labels, "user_id"].unique())
        assert not (train_participants_fold & test_participants_fold), "grouped-CV participant leakage"

        print(f"\n=== Grouped-CV fold {fold_i} (TreeAvg, n_k/N-weighted) ===")
        r = run_treeavg_fold(train_participants_fold, test_idx_labels, labeled, feature_cols, y)
        fold_auroc = float(roc_auc_score(r["y_test"], r["test_pred"]))
        print(f"fold {fold_i}: AUROC={fold_auroc:.4f} best_round={r['best_round']} n_test={r['n_test_rows']} weights={[round(w, 4) for w in r['client_weights_n_k_over_N']]}")
        fold_results.append({"fold": fold_i, "auroc": fold_auroc, "best_round": r["best_round"], "best_val_auroc": r["best_val_auroc"],
                              "client_n_participants": r["client_n_participants"], "client_n_rows": r["client_n_rows"],
                              "client_weights_n_k_over_N": r["client_weights_n_k_over_N"], "n_test_rows": r["n_test_rows"]})
        all_y.append(r["y_test"])
        all_pred.append(r["test_pred"])

    y_pooled = np.concatenate(all_y)
    pred_pooled = np.concatenate(all_pred)
    pooled_metrics = extended_metrics("tree_avg_grouped_cv_pooled_OOF", y_pooled, pred_pooled)

    results = {
        "experiment": "F -- grouped cross-validation confirmation",
        "n_splits": N_SPLITS, "client_partition_seed": CLIENT_PARTITION_SEED, "n_rounds": N_ROUNDS_F,
        "weighting_scheme": "sample-count-weighted n_k/N, recomputed per fold",
        "fold_results": fold_results,
        "pooled_oof_metrics": pooled_metrics,
        "fixed_split_treeavg_reference_auroc_mean": None,  # filled by caller/report from step5 results if desired
        "cited_centralized_lightgbm_5fold_pooled_oof_reference_auroc": C.ARM_B_CENTRALIZED_LGBM_DART_REFERENCE["auroc"],
        "runtime_seconds": time.time() - t0,
    }
    out_path = SCRIPT_DIR / "psyche_d_fedavg_vs_fedtree_step5_experiment_f_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")
    print(f"Pooled 3-fold grouped-CV TreeAvg AUROC: {pooled_metrics['auroc']:.4f} [{pooled_metrics['auroc_ci'][0]:.4f},{pooled_metrics['auroc_ci'][1]:.4f}]")
    print(f"Per-fold AUROCs: {[r['auroc'] for r in fold_results]}")
    print(f"runtime={results['runtime_seconds']:.1f}s")
    return results


if __name__ == "__main__":
    run()
