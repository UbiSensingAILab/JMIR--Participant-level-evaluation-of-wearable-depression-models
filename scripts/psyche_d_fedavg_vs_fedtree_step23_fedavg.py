#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. `research_datasets/` is
# gitignored; local research data, never committed to this repo.
"""Steps 2/3 of `.planning/debug/psyche-d-fedavg-vs-fedtree.md`: real FedAvg
simulation over the SAME fixed train/val/test split cached by Step 1.

Protocol (per task instruction, verbatim requirements enforced):
  - Participants split into train/val/test FIRST (loaded from Step 1's cache,
    never rebuilt here) -- val/test participants NEVER appear in any client.
  - Whole-participant K=5 client partition of the train pool ONLY, 3 different
    partition seeds (tests partition-dependence without a hyperparameter hunt).
  - NO centralized pretraining: FedAvg round 0 starts from a COMMON randomly
    initialized MLP (`M.new_model(seed=M.SEED)`), distributed to all clients.
  - Federated-safe preprocessing: mean/std computed by aggregating each
    client's local (count, sum, sumsq) -- never raw rows, never val/test rows.
  - Local-only (no-aggregation) per-client models reported alongside FedAvg,
    same architecture, same total local-epoch compute budget.
  - Global model selected by VALIDATION AUROC across rounds (never test),
    final number is the selected round's global model evaluated on test.
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
import psyche_d_fedavg_vs_fedtree_common as C  # noqa: E402
import psyche_d_fedavg_vs_fedtree_mlp_model as M  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
SPLIT_CACHE_PATH = SCRIPT_DIR / "psyche_d_fedavg_vs_fedtree_split_cache.pkl"
N_ROUNDS = 40
LOCAL_EPOCHS = 2
LR = 1e-3


def load_split_and_frame():
    cache = pickle.loads(SPLIT_CACHE_PATH.read_bytes())
    labeled, feature_cols_fresh, y = C.load_arm_b_frame()
    assert feature_cols_fresh == cache["feature_cols"], "feature construction drifted since Step 1 -- must match exactly"
    train_idx = labeled.index[labeled["user_id"].isin(cache["train_participants"])]
    val_idx = labeled.index[labeled["user_id"].isin(cache["val_participants"])]
    test_idx = labeled.index[labeled["user_id"].isin(cache["test_participants"])]
    return cache, labeled, cache["feature_cols"], y, train_idx, val_idx, test_idx


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
    client_n_participants = [len(s) for s in sites]
    client_n_rows = [len(d) for d in client_dfs]
    for i, d in enumerate(client_dfs):
        assert d["user_id"].isin(sites[i]).all()
        assert y.loc[d.index].nunique() == 2, f"client {i} has only one class -- cannot train/evaluate locally"

    # Federated-safe preprocessing: aggregate sufficient stats ACROSS clients
    # (K groups) -- never a centralized pooled fit, never touches val/test.
    stats = C.federated_sufficient_stats(client_dfs, feature_cols)
    mean, std = stats["mean"], stats["std"]
    X_val = C.apply_impute_scale(val_df, feature_cols, mean, std)
    X_test = C.apply_impute_scale(test_df, feature_cols, mean, std)
    client_X = [C.apply_impute_scale(d, feature_cols, mean, std) for d in client_dfs]
    client_y = [y.loc[d.index].to_numpy() for d in client_dfs]

    # --- Local-only (no-aggregation) baseline: each client trains solely on
    # its own data for the SAME total local-epoch budget FedAvg would use
    # (N_ROUNDS * LOCAL_EPOCHS), starting from the SAME common random init. ---
    local_only_metrics = []
    for cid in range(C.N_CLIENTS):
        local_model = M.new_model(seed=M.SEED)
        M.train_local(local_model, client_X[cid], client_y[cid], epochs=N_ROUNDS * LOCAL_EPOCHS, lr=LR, seed=1000 + cid)
        proba = M.predict_proba(local_model, X_test)
        local_only_metrics.append(C.compute_metrics(f"local_only_client{cid}_seed{partition_seed} (test)", y_test, proba))

    # --- Real FedAvg: common init -> distribute -> local train -> aggregate -> repeat ---
    global_model = M.new_model(seed=M.SEED)
    best_val_auroc = -1.0
    best_state = None
    best_round = -1
    best_val_proba_snapshot = None
    round_history = []
    for rnd in range(N_ROUNDS):
        global_state = M.get_state(global_model)
        client_states = []
        for cid in range(C.N_CLIENTS):
            local_model = M.SmallMLP()
            M.set_state(local_model, global_state)
            M.train_local(local_model, client_X[cid], client_y[cid], epochs=LOCAL_EPOCHS, lr=LR, seed=rnd * 100 + cid)
            client_states.append(M.get_state(local_model))
        new_global_state = M.fedavg_aggregate(client_states, client_n_rows)
        M.set_state(global_model, new_global_state)
        val_proba = M.predict_proba(global_model, X_val)
        val_auroc = float(roc_auc_score(y_val, val_proba))
        round_history.append({"round": rnd, "val_auroc": val_auroc})
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = M.get_state(global_model)
            best_round = rnd
            best_val_proba_snapshot = val_proba.copy()

    M.set_state(global_model, best_state)
    test_proba_fedavg = M.predict_proba(global_model, X_test)
    fedavg_metrics = C.compute_metrics(f"fedavg_mlp_seed{partition_seed} (test)", y_test, test_proba_fedavg)

    return {
        "partition_seed": partition_seed,
        "disjointness": disjoint,
        "client_n_participants": client_n_participants,
        "client_n_rows": client_n_rows,
        "local_only_per_client_metrics": local_only_metrics,
        "local_only_mean_auroc": float(np.mean([m["auroc"] for m in local_only_metrics])),
        "local_only_std_auroc": float(np.std([m["auroc"] for m in local_only_metrics])),
        "fedavg_best_round": best_round,
        "fedavg_best_val_auroc": best_val_auroc,
        "fedavg_n_rounds_run": N_ROUNDS,
        "fedavg_local_epochs_per_round": LOCAL_EPOCHS,
        "fedavg_test_metrics": fedavg_metrics,
        # Added for Step 5's Experiment B -- purely additive provenance.
        "y_test": y_test.tolist(),
        "fedavg_test_predictions": test_proba_fedavg.tolist(),
        # Added for the psyche-d-treeavg-audit-extend session's Section 13
        # (validation-selected decision threshold) -- purely additive.
        "y_val": y_val.tolist(),
        "fedavg_val_predictions": best_val_proba_snapshot.tolist(),
    }


def run() -> dict:
    t0 = time.time()
    cache, labeled, feature_cols, y, train_idx, val_idx, test_idx = load_split_and_frame()

    per_seed_results = []
    for seed in C.PARTITION_SEEDS:
        print(f"\n=== Partition seed {seed} ===")
        per_seed_results.append(run_one_seed(seed, cache, labeled, feature_cols, y, train_idx, val_idx, test_idx))

    fedavg_aurocs = [r["fedavg_test_metrics"]["auroc"] for r in per_seed_results]
    local_only_aurocs = [r["local_only_mean_auroc"] for r in per_seed_results]

    results = {
        "step": "2/3",
        "description": "Real FedAvg simulation (common random init, K=5 whole-participant clients, 3 partition seeds) + local-only no-aggregation baseline, on Step 1's fixed train/val/test split.",
        "n_clients": C.N_CLIENTS,
        "partition_seeds": C.PARTITION_SEEDS,
        "fedavg_hyperparams": {"n_rounds": N_ROUNDS, "local_epochs_per_round": LOCAL_EPOCHS, "lr": LR},
        "per_seed_results": per_seed_results,
        "partition_dependence": {
            "fedavg_auroc_mean": float(np.mean(fedavg_aurocs)),
            "fedavg_auroc_std": float(np.std(fedavg_aurocs)),
            "fedavg_auroc_by_seed": dict(zip(C.PARTITION_SEEDS, fedavg_aurocs)),
            "local_only_auroc_mean": float(np.mean(local_only_aurocs)),
            "local_only_auroc_std": float(np.std(local_only_aurocs)),
            "local_only_auroc_by_seed": dict(zip(C.PARTITION_SEEDS, local_only_aurocs)),
        },
        "runtime_seconds": time.time() - t0,
    }

    json_path = SCRIPT_DIR / "psyche_d_fedavg_vs_fedtree_step23_results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {json_path}")
    print(f"\nFedAvg AUROC by seed: {results['partition_dependence']['fedavg_auroc_by_seed']}")
    print(f"FedAvg AUROC mean={results['partition_dependence']['fedavg_auroc_mean']:.4f} std={results['partition_dependence']['fedavg_auroc_std']:.4f}")
    print(f"Local-only mean AUROC by seed: {results['partition_dependence']['local_only_auroc_by_seed']}")
    return results


if __name__ == "__main__":
    run()
