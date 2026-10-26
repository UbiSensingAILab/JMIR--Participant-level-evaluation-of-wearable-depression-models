#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. `research_datasets/` is
# gitignored; local research data, never committed to this repo.
"""Step 5 -- the CORRECTED "TreeAvg" federated-GBDT experiment, distinct
from Step 4's FedXgbBagging. This is a from-scratch rewrite (per explicit
task instruction) superseding an earlier same-day uncommitted version of
this file that used an undisclosed UNIFORM 1/K per-client weight.

DEFINITION (the one remaining ambiguity in this definition -- uniform 1/K
vs. sample-count-weighted n_k/N per-client weighting -- was raised to the
user as an explicit checkpoint and resolved by their answer: "Sample-count-
weighted n_k/N (Recommended)", matching McMahan et al. (2017)'s actual
FedAvg rule and this project's OWN existing convention in
`psyche_d_fedavg_vs_fedtree_mlp_model.py:fedavg_aggregate()`
(`agg[key] += state[key] * (n / total_n)`). See the toy-example script
`psyche_d_fedavg_vs_fedtree_step5_toy_example.py` for a hand-checkable
proof this is NOT equivalent to bagging or to averaging independent final
local models.):

  At round r, each of the K=5 clients grows exactly ONE new tree h_{k,r}
  continuing from the CURRENT SHARED TreeAvg global margin (identical
  client-side mechanic to Step 4's FedXgbBagging -- see `local_boost` in
  that file). The server does NOT concatenate the K trees onto the
  ensemble at full weight (that is what `flwr.server.strategy
  .fedxgb_bagging.aggregate` does, and this script never calls it). Instead
  the server combines them by taking the SAMPLE-COUNT-WEIGHTED MEAN of
  their margin outputs:

      w_k               = n_k / N   (n_k = client k's TRAINING ROW count
                                      under the CURRENT partition seed;
                                      N = sum_k n_k for that SAME seed --
                                      recomputed per seed, not a fixed
                                      constant across the 3 required seeds)
      contribution_r(x) = sum_{k=1}^{K} w_k * h_{k,r}(x)
      margin_T(x)        = sum_{r=1}^{T} contribution_r(x)
      prediction(x)      = sigmoid(base_margin + margin_T(x))

  This is implemented EXPLICITLY (never via XGBoost JSON leaf-surgery):
  the global model is a flat list of (round, client_id, Booster) triples,
  each with an EXPLICIT weight w_k = n_k/N, and every prediction is
  computed by summing w_k * tree.predict(D, output_margin=True) over all
  stored trees. This keeps the averaging mechanism fully auditable.

  Per-round, EVERY client (not just the tree-growers) sees the SAME
  updated (weighted-averaged) global margin as its base_margin for the
  next round -- this "synchronized global state" is what makes it a
  genuine round-by-round FedAvg-style procedure rather than one-shot
  ensembling.

Protocol (identical to Step 4, so the comparison is meaningful): same
fixed split, same 5-client partitions/seeds, same XGB hyperparameter
family, same class-imbalance handling, same validation-based best-round
selection, same test set, never touched during development.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import psyche_d_fedavg_vs_fedtree_common as C  # noqa: E402
from psyche_d_fedavg_vs_fedtree_step4_fed_xgb_bagging import (  # noqa: E402
    NUM_LOCAL_ROUND,
    N_ROUNDS,
    XGB_BASE_PARAMS,
    load_split_and_frame,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DECISION_THRESHOLD = 0.5


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def extended_metrics(name: str, y_true: np.ndarray, y_score: np.ndarray, threshold: float = DECISION_THRESHOLD) -> dict:
    """Extends `C.compute_metrics` (AUROC/AUPRC/accuracy/balanced-acc/sens/
    spec/CI/confusion, already computed by the shared helper) with the
    additional metrics this task's Section 3 requires that the shared
    helper does not already provide: F1, Brier score, and an explicit
    confusion matrix at the pre-specified decision threshold (0.5, the
    same threshold `C.compute_metrics` itself already uses for its own
    tp/fp/tn/fn -- reported again here under the task's own field names
    for clarity)."""
    base = C.compute_metrics(name, y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    base["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    base["brier_score"] = float(brier_score_loss(y_true, y_score))
    base["decision_threshold"] = threshold
    base["confusion_matrix_at_threshold"] = {
        "tp": base["tp"], "fp": base["fp"], "tn": base["tn"], "fn": base["fn"],
    }
    # sanity cross-check against sklearn's own accuracy/balanced_accuracy at
    # the same threshold (the shared helper already uses >=0.5 internally,
    # so these should reconcile exactly)
    assert abs(base["accuracy"] - accuracy_score(y_true, y_pred)) < 1e-9
    assert abs(base["balanced_accuracy"] - balanced_accuracy_score(y_true, y_pred)) < 1e-9
    return base


def run_one_seed(partition_seed: int, cache, labeled, feature_cols, y, train_idx, val_idx, test_idx) -> dict:
    train_df = labeled.loc[train_idx]
    val_df = labeled.loc[val_idx]
    test_df = labeled.loc[test_idx]
    y_val = y.loc[val_idx].to_numpy()
    y_test = y.loc[test_idx].to_numpy()

    sites = C.assign_clients(cache["train_participants"], k=C.N_CLIENTS, seed=partition_seed)
    disjoint = C.verify_disjointness(
        {"train_participants": cache["train_participants"], "val_participants": cache["val_participants"], "test_participants": cache["test_participants"]},
        sites,
    )
    client_dfs = [train_df[train_df["user_id"].isin(s)] for s in sites]

    total_pos = sum(int((y.loc[d.index] == 1).sum()) for d in client_dfs)
    total_neg = sum(int((y.loc[d.index] == 0).sum()) for d in client_dfs)
    scale_pos_weight = total_neg / max(total_pos, 1)
    params = dict(XGB_BASE_PARAMS, scale_pos_weight=scale_pos_weight)
    K = C.N_CLIENTS

    # --- Sample-count-weighted (n_k/N) TreeAvg weights, RECOMPUTED for this
    # partition seed (client row counts vary by seed, so N and every w_k are
    # seed-dependent, not a fixed constant reused across seeds 1/2/3). ---
    client_n_rows = np.array([len(d) for d in client_dfs], dtype=np.float64)
    N_total_rows = float(client_n_rows.sum())
    client_weights = client_n_rows / N_total_rows
    assert abs(client_weights.sum() - 1.0) < 1e-9, "client weights must sum to 1"

    # Persistent per-client training DMatrices (label attached; base_margin
    # will be updated in place every round via set_base_margin).
    client_dmatrices = []
    client_margin = []  # train-side cumulative TreeAvg margin, per client
    for d in client_dfs:
        yc = y.loc[d.index].to_numpy()
        assert len(np.unique(yc)) == 2, "client has only one class -- cannot train"
        Xc = d[feature_cols].to_numpy(dtype=np.float64)
        dm = xgb.DMatrix(Xc, label=yc, missing=np.nan)
        dm.set_base_margin(np.zeros(len(yc)))
        client_dmatrices.append(dm)
        client_margin.append(np.zeros(len(yc)))
    client_X = [d[feature_cols].to_numpy(dtype=np.float64) for d in client_dfs]

    X_val = val_df[feature_cols].to_numpy(dtype=np.float64)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float64)
    d_val_eval = xgb.DMatrix(X_val, missing=np.nan)  # base_margin left at 0 -> pure per-tree margins
    d_test_eval = xgb.DMatrix(X_test, missing=np.nan)
    d_client_eval = [xgb.DMatrix(Xc, missing=np.nan) for Xc in client_X]  # zero-base-margin, for per-tree contribution on own client rows

    val_margin = np.zeros(len(y_val))
    test_margin = np.zeros(len(y_test))

    best_val_auroc = -1.0
    best_round = -1
    best_val_margin_snapshot = None
    best_test_margin_snapshot = None
    round_history = []
    all_trees_bytes = []  # for approx. communication-volume accounting
    n_trees_per_client = [0] * K
    # Section 14 (psyche-d-treeavg-audit-extend): REAL per-round payload --
    # upload = the K clients' real per-round tree JSON sizes (the actual
    # bytes fed into the weighted-sum combination below); download = the
    # server broadcasting that SAME round's K tree JSONs back to every
    # client (so each client can compute its own updated running margin) --
    # this is CONSTANT size per round (not the whole accumulated ensemble),
    # since every client maintains its own running `client_margin` state and
    # only needs this round's K new trees, never the full history again.
    upload_bytes_per_round = []
    download_bytes_per_round = []

    for rnd in range(1, N_ROUNDS + 1):
        round_boosters = []
        round_tree_bytes = []
        for k in range(K):
            client_dmatrices[k].set_base_margin(client_margin[k])
            bst = xgb.train(params, client_dmatrices[k], num_boost_round=NUM_LOCAL_ROUND)
            round_boosters.append(bst)
            n_trees_per_client[k] += 1
            tree_bytes = len(bst.save_raw("json"))
            all_trees_bytes.append(tree_bytes)
            round_tree_bytes.append(tree_bytes)
        upload_bytes_per_round.append(sum(round_tree_bytes))
        download_bytes_per_round.append(sum(round_tree_bytes) * K)  # server rebroadcasts this round's K trees to all K clients

        # This round's pure (zero-base-margin) tree contributions, per client, on
        # (a) each client's own train rows, (b) val, (c) test -- combined via the
        # n_k/N SAMPLE-COUNT-WEIGHTED sum (NOT a uniform 1/K mean).
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

        val_pred = sigmoid(val_margin)
        val_auroc = float(roc_auc_score(y_val, val_pred))
        # Section 3 (psyche-d-treeavg-audit-extend) diagnostic: also track val
        # AUPRC and val log-loss every round -- purely additive (recorded,
        # never used to influence training/stopping, which remains val AUROC
        # only, exactly as before).
        val_auprc = float(average_precision_score(y_val, val_pred))
        eps = 1e-12
        val_logloss = float(-np.mean(y_val * np.log(np.clip(val_pred, eps, 1 - eps)) + (1 - y_val) * np.log(np.clip(1 - val_pred, eps, 1 - eps))))
        round_history.append({"round": rnd, "val_auroc": val_auroc, "val_auprc": val_auprc, "val_logloss": val_logloss, "n_trees_total": K * rnd})
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_round = rnd
            best_val_margin_snapshot = val_margin.copy()
            best_test_margin_snapshot = test_margin.copy()

    test_pred = sigmoid(best_test_margin_snapshot)
    tree_avg_metrics = extended_metrics(f"tree_avg_seed{partition_seed} (test)", y_test, test_pred)

    approx_bytes_per_tree = float(np.mean(all_trees_bytes))
    total_trees_communicated = K * N_ROUNDS  # each client uploads 1 tree/round; server broadcasts the round's K trees (or equiv. weighted-avg representation) back
    approx_comm_bytes_total = approx_bytes_per_tree * total_trees_communicated * 2  # x2: upload + broadcast-back

    return {
        "partition_seed": partition_seed,
        "disjointness": disjoint,
        "client_n_participants": [len(s) for s in sites],
        "client_n_rows": [int(n) for n in client_n_rows.tolist()],
        "client_weights_n_k_over_N": client_weights.tolist(),
        "N_total_rows_this_seed": N_total_rows,
        "scale_pos_weight_federated_agg": scale_pos_weight,
        "best_round": best_round,
        "best_val_auroc": best_val_auroc,
        "n_trees_per_client": n_trees_per_client,
        "final_n_trees_total": K * N_ROUNDS,
        "n_communication_rounds": N_ROUNDS,
        "approx_bytes_per_tree_json": approx_bytes_per_tree,
        "approx_total_communication_bytes": approx_comm_bytes_total,
        "tree_avg_test_metrics": tree_avg_metrics,
        "y_test": y_test.tolist(),
        "tree_avg_test_predictions": test_pred.tolist(),
        # Added for the psyche-d-treeavg-audit-extend session:
        # Section 3 (validation-curve diagnostic) -- purely additive, the
        # per-round values were already being computed above, just exposed.
        "round_history": round_history,
        # Section 13 (validation-selected decision threshold) -- purely additive.
        "y_val": y_val.tolist(),
        "tree_avg_val_predictions": sigmoid(best_val_margin_snapshot).tolist(),
        # Section 14 (real, independently measured communication payload,
        # per-round upload/download -- same measurement methodology now also
        # applied to FedXgbBagging in step4.py for an apples-to-apples,
        # independently-measured comparison) -- purely additive.
        "real_measured_communication": {
            "upload_bytes_per_round": upload_bytes_per_round,
            "download_bytes_per_round": download_bytes_per_round,
            "total_upload_bytes": int(sum(upload_bytes_per_round)),
            "total_download_bytes": int(sum(download_bytes_per_round)),
            "total_communication_bytes": int(sum(upload_bytes_per_round) + sum(download_bytes_per_round)),
            "note": "upload = sum of all K clients' real per-round tree JSON sizes; download = the SAME round's K trees rebroadcast to all K clients (constant size per round, since every client keeps its own running margin and only ever needs the current round's new trees, never the full history) -- structurally different from FedXgbBagging's download, which resends the WHOLE growing ensemble every round.",
        },
    }


def run() -> dict:
    t0 = time.time()
    cache, labeled, feature_cols, y, train_idx, val_idx, test_idx = load_split_and_frame()

    per_seed_results = []
    for seed in C.PARTITION_SEEDS:
        print(f"\n=== TreeAvg partition seed {seed} ===")
        per_seed_results.append(run_one_seed(seed, cache, labeled, feature_cols, y, train_idx, val_idx, test_idx))

    aurocs = [r["tree_avg_test_metrics"]["auroc"] for r in per_seed_results]
    auprcs = [r["tree_avg_test_metrics"]["auprc"] for r in per_seed_results]
    accs = [r["tree_avg_test_metrics"]["accuracy"] for r in per_seed_results]
    f1s = [r["tree_avg_test_metrics"]["f1"] for r in per_seed_results]
    sens = [r["tree_avg_test_metrics"]["sensitivity"] for r in per_seed_results]
    specs = [r["tree_avg_test_metrics"]["specificity"] for r in per_seed_results]
    bal_accs = [r["tree_avg_test_metrics"]["balanced_accuracy"] for r in per_seed_results]
    briers = [r["tree_avg_test_metrics"]["brier_score"] for r in per_seed_results]

    results = {
        "step": 5,
        "description": "TreeAvg: round-wise SAMPLE-COUNT-WEIGHTED (n_k/N) mean-of-margins federated GBDT (server never calls flwr's fedxgb_bagging.aggregate; combination is an explicit weighted sum of per-client-per-round tree margins, weight n_k/N each, recomputed per partition seed). NOT a rename of FedXgbBagging, and NOT a uniform-1/K average -- see module docstring and the toy-example script for the mathematical distinction and the weighting-scheme decision.",
        "algorithm_disclosure": {
            "definition": "global_margin(x) = sum over rounds r, clients k of (n_k/N) * tree_{r,k}.predict(x, output_margin=True); prediction = sigmoid(base_margin + global_margin(x)); n_k = client k's training row count for the CURRENT partition seed, N = sum_k n_k for that same seed",
            "weighting_scheme_decision": "sample-count-weighted n_k/N (McMahan et al. 2017 FedAvg convention, matching this project's own mlp_model.py:fedavg_aggregate()) -- chosen by explicit user decision over uniform 1/K at the Part-1 checkpoint",
            "client_side_mechanic": "identical to Step 4's local_boost -- each client grows exactly 1 new tree per round from the CURRENT SHARED (weighted-averaged) global margin",
            "server_side_combination": "sample-count-weighted (n_k/N) sum of the K clients' same-round tree outputs (NOT concatenation, NOT uniform 1/K, NOT XGBoost JSON leaf-value surgery)",
            "toy_example_verification": "scripts/psyche_d_fedavg_vs_fedtree_step5_toy_example.py / _results.json -- proves TreeAvg != naive concatenation-of-same-trees, != faithful independent bagging ensemble, and != averaging-two-independently-trained-final-models' predictions",
        },
        "n_clients": C.N_CLIENTS,
        "partition_seeds": C.PARTITION_SEEDS,
        "hyperparams": {"num_local_round_per_client_per_communication_round": NUM_LOCAL_ROUND, "n_rounds": N_ROUNDS, "xgb_params": XGB_BASE_PARAMS},
        "per_seed_results": per_seed_results,
        "partition_dependence": {
            "auroc_mean": float(np.mean(aurocs)), "auroc_std": float(np.std(aurocs)), "auroc_by_seed": dict(zip(C.PARTITION_SEEDS, aurocs)),
            "auprc_mean": float(np.mean(auprcs)), "auprc_std": float(np.std(auprcs)),
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "sensitivity_mean": float(np.mean(sens)), "sensitivity_std": float(np.std(sens)),
            "specificity_mean": float(np.mean(specs)), "specificity_std": float(np.std(specs)),
            "balanced_accuracy_mean": float(np.mean(bal_accs)), "balanced_accuracy_std": float(np.std(bal_accs)),
            "brier_score_mean": float(np.mean(briers)), "brier_score_std": float(np.std(briers)),
        },
        "runtime_seconds": time.time() - t0,
    }

    json_path = SCRIPT_DIR / "psyche_d_fedavg_vs_fedtree_step5_results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {json_path}")
    print(f"\nTreeAvg AUROC by seed: {results['partition_dependence']['auroc_by_seed']}")
    print(f"mean={results['partition_dependence']['auroc_mean']:.4f} std={results['partition_dependence']['auroc_std']:.4f}")
    print(f"runtime={results['runtime_seconds']:.1f}s")
    return results


if __name__ == "__main__":
    run()
