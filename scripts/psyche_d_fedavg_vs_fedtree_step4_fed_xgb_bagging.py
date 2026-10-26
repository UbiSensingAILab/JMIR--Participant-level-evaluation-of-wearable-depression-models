#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. `research_datasets/` is
# gitignored; local research data, never committed to this repo.
"""Step 4 of `.planning/debug/psyche-d-fedavg-vs-fedtree.md`: a genuinely-
verified federated gradient-boosted-tree implementation, on the SAME fixed
train/val/test split and SAME client assignments as Steps 2/3.

HONESTY DISCLOSURE (per task's explicit, non-negotiable requirement --
"verify what exact algorithm/implementation is actually being used... do not
fabricate compatibility with LightGBM-DART if the implementation is actually
a different GBDT formulation"):

  1. Real `pip install fedtree` was re-attempted fresh in this session and
     STILL fails to build on this macOS environment (missing prebuilt wheel,
     needs a from-source CMake build of `libFedTree.dylib`) -- identical
     failure to the prior FL-experiment-design session. See Evidence log.
  2. `nvflare` and `flwr[simulation]` were both confirmed genuinely
     installable. This script uses Flower (flwr 1.38.0) -- specifically its
     REAL, UNMODIFIED `flwr.server.strategy.fedxgb_bagging.aggregate()`
     function (imported directly from the installed library, not
     reimplemented) to perform server-side tree aggregation.
  3. ALGORITHM: this is "FedXgbBagging" -- Flower's own name for it. It is
     TREE-LEVEL BAGGING AGGREGATION: each client grows exactly ONE new
     XGBoost tree per communication round on its own local data (continuing
     from the current global ensemble), and the server AGGREGATES by
     concatenating every client's newly-grown tree onto a shared, growing
     global ensemble (verified by reading the real aggregate() source: it
     JSON-decodes each client's delta booster and appends its trees to the
     previous global booster's tree list).
  4. This is NOT FedTree/SecureBoost's per-feature-bin gradient/Hessian
     histogram aggregation (which cooperatively decides EACH SPLIT from
     aggregated statistics before any tree exists) -- it is a fundamentally
     different, coarser-grained federation mechanism (whole trees, built
     independently per client, then unioned). It is also XGBoost, NOT
     LightGBM -- no claim of LightGBM-DART compatibility is made anywhere
     in this script or its outputs.
  5. The CLIENT-SIDE boosting step (loading the global model, calling
     `Booster.update()` for `num_local_round` additional rounds, then
     slicing out just the newly-added tree(s) via `Booster.__getitem__`) is
     re-implemented here from Flower's own PUBLISHED reference example
     (`adap/flower`, `examples/xgboost-comprehensive/xgboost_comprehensive/
     client_app.py`, function `_local_boost`, inspected via GitHub on
     2026-09-28) -- that logic lives in Flower's example/app code, not in
     the installed `flwr` package itself, so it cannot be imported verbatim;
     it is reproduced faithfully (same `Booster.update`/slicing calls, same
     `num_parallel_tree=1` convention) rather than invented from scratch.
     The SERVER-side aggregation math (the part that actually defines "what
     bagging means" here) IS the real, unmodified library function.
  6. Full round-1-through-N pattern, class-imbalance handling via
     `scale_pos_weight` computed from FEDERATED-SAFE aggregated per-client
     positive/negative counts (never raw rows, never val/test rows) --
     mirroring this session's federated-safe preprocessing principle.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from flwr.server.strategy.fedxgb_bagging import aggregate as flwr_fedxgb_aggregate
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import psyche_d_fedavg_vs_fedtree_common as C  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
SPLIT_CACHE_PATH = SCRIPT_DIR / "psyche_d_fedavg_vs_fedtree_split_cache.pkl"
NUM_LOCAL_ROUND = 1     # exactly 1 new tree per client per round -- Flower's own documented FedXgbBagging convention
N_ROUNDS = 30            # 5 clients x 1 tree x 30 rounds = 150 total trees in the final global ensemble (matches LightGBM-DART n_estimators=150 for comparability)
XGB_BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "eta": 0.1,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "num_parallel_tree": 1,
    "tree_method": "hist",
}


def load_split_and_frame():
    cache = pickle.loads(SPLIT_CACHE_PATH.read_bytes())
    labeled, feature_cols_fresh, y = C.load_arm_b_frame()
    assert feature_cols_fresh == cache["feature_cols"], "feature construction drifted since Step 1 -- must match exactly"
    train_idx = labeled.index[labeled["user_id"].isin(cache["train_participants"])]
    val_idx = labeled.index[labeled["user_id"].isin(cache["val_participants"])]
    test_idx = labeled.index[labeled["user_id"].isin(cache["test_participants"])]
    return cache, labeled, cache["feature_cols"], y, train_idx, val_idx, test_idx


def local_boost(bst_input: xgb.Booster, num_local_round: int, dtrain: xgb.DMatrix) -> xgb.Booster:
    """Faithful reproduction of Flower's own `client_app.py:_local_boost`
    (bagging branch) -- continues boosting the LOADED global model for
    `num_local_round` additional rounds, then slices out ONLY the newly
    added tree(s) to return as this client's per-round contribution."""
    for _ in range(num_local_round):
        bst_input.update(dtrain, bst_input.num_boosted_rounds())
    n = bst_input.num_boosted_rounds()
    return bst_input[n - num_local_round: n]


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

    # XGBoost handles missing values natively (no SimpleImputer needed) --
    # raw features (with real NaNs) passed straight into DMatrix(missing=nan).
    # Federated-safe class-imbalance handling: aggregate per-client positive/
    # negative COUNTS only (never raw rows) to compute a single global
    # scale_pos_weight, applied identically at every client.
    total_pos = sum(int((y.loc[d.index] == 1).sum()) for d in client_dfs)
    total_neg = sum(int((y.loc[d.index] == 0).sum()) for d in client_dfs)
    scale_pos_weight = total_neg / max(total_pos, 1)
    params = dict(XGB_BASE_PARAMS, scale_pos_weight=scale_pos_weight)

    client_dmatrices = []
    for d in client_dfs:
        yc = y.loc[d.index].to_numpy()
        assert len(np.unique(yc)) == 2, "client has only one class -- cannot train"
        Xc = d[feature_cols].to_numpy(dtype=np.float64)
        client_dmatrices.append(xgb.DMatrix(Xc, label=yc, missing=np.nan))

    X_val = val_df[feature_cols].to_numpy(dtype=np.float64)
    dval = xgb.DMatrix(X_val, label=y_val, missing=np.nan)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float64)
    dtest = xgb.DMatrix(X_test, label=y_test, missing=np.nan)

    global_model_bytes = None
    best_val_auroc = -1.0
    best_model_bytes = None
    best_round = -1
    best_val_pred_snapshot = None
    round_history = []
    # Section 14 (psyche-d-treeavg-audit-extend): independently measure the
    # REAL serialized payload this algorithm actually transfers each round --
    # client->server upload is each client's per-round delta-booster JSON
    # (`round_client_deltas`, exactly what's passed into `flwr_fedxgb_aggregate`
    # client-by-client); server->client download is the resulting
    # `global_model_bytes` broadcast back to every client so its next-round
    # `local_boost` continues from the current global ensemble. Both measured
    # directly from the actual bytes objects already produced by the real
    # unmodified `flwr_fedxgb_aggregate` call -- NOT inferred/reused from
    # TreeAvg's tree size (that reuse, disclosed in Experiment E, is the
    # specific shortcut this instrumentation corrects).
    upload_bytes_per_round = []
    download_bytes_per_round = []

    for rnd in range(1, N_ROUNDS + 1):
        round_client_deltas = []
        for dtrain in client_dmatrices:
            if global_model_bytes is None:
                bst = xgb.train(params, dtrain, num_boost_round=NUM_LOCAL_ROUND)
                delta = bst  # round 1: entire (1-tree) local model is "new"
            else:
                bst = xgb.Booster(params=params)
                bst.load_model(bytearray(global_model_bytes))
                delta = local_boost(bst, NUM_LOCAL_ROUND, dtrain)
            round_client_deltas.append(delta.save_raw("json"))

        upload_bytes_per_round.append(sum(len(d) for d in round_client_deltas))
        for delta_bytes in round_client_deltas:
            global_model_bytes = flwr_fedxgb_aggregate(global_model_bytes, delta_bytes)
        # server broadcasts the updated global ensemble back to all K clients
        # (each client's next-round local_boost loads this exact byte string)
        download_bytes_per_round.append(len(global_model_bytes) * len(client_dmatrices))

        eval_bst = xgb.Booster(params=params)
        eval_bst.load_model(bytearray(global_model_bytes))
        val_pred = eval_bst.predict(dval)
        val_auroc = float(roc_auc_score(y_val, val_pred))
        round_history.append({"round": rnd, "val_auroc": val_auroc, "n_trees": eval_bst.num_boosted_rounds()})
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_model_bytes = bytes(global_model_bytes)
            best_round = rnd
            best_val_pred_snapshot = val_pred.copy()

    final_bst = xgb.Booster(params=params)
    final_bst.load_model(bytearray(best_model_bytes))
    test_pred = final_bst.predict(dtest)
    fed_xgb_metrics = C.compute_metrics(f"fed_xgb_bagging_seed{partition_seed} (test)", y_test, test_pred)

    return {
        "partition_seed": partition_seed,
        "disjointness": disjoint,
        "client_n_participants": [len(s) for s in sites],
        "client_n_rows": [len(d) for d in client_dfs],
        "scale_pos_weight_federated_agg": scale_pos_weight,
        "best_round": best_round,
        "best_val_auroc": best_val_auroc,
        "final_n_trees": final_bst.num_boosted_rounds(),
        "fed_xgb_bagging_test_metrics": fed_xgb_metrics,
        # Added for Step 5's Experiment B -- purely additive provenance.
        "y_test": y_test.tolist(),
        "fed_xgb_bagging_test_predictions": test_pred.tolist(),
        # Added for the psyche-d-treeavg-audit-extend session's Section 13
        # (validation-selected decision threshold) -- purely additive.
        "y_val": y_val.tolist(),
        "fed_xgb_bagging_val_predictions": best_val_pred_snapshot.tolist(),
        # Added for the same session's Section 14 (real, independently
        # measured communication payload, correcting the Experiment E
        # disclosed shortcut of reusing TreeAvg's bytes/tree as a proxy) --
        # purely additive, no existing control-flow touched.
        "real_measured_communication": {
            "upload_bytes_per_round": upload_bytes_per_round,
            "download_bytes_per_round": download_bytes_per_round,
            "total_upload_bytes": int(sum(upload_bytes_per_round)),
            "total_download_bytes": int(sum(download_bytes_per_round)),
            "total_communication_bytes": int(sum(upload_bytes_per_round) + sum(download_bytes_per_round)),
            "note": "upload = sum of all K clients' real per-round delta-booster JSON sizes (the exact bytes fed into flwr_fedxgb_aggregate); download = the resulting accumulated global_model_bytes (whole growing ensemble) broadcast to all K clients -- this GROWS every round since FedXgbBagging's global model is a monotonically growing tree ensemble, unlike TreeAvg's constant-size per-round update.",
        },
    }


def run() -> dict:
    t0 = time.time()
    cache, labeled, feature_cols, y, train_idx, val_idx, test_idx = load_split_and_frame()

    per_seed_results = []
    for seed in C.PARTITION_SEEDS:
        print(f"\n=== Partition seed {seed} ===")
        per_seed_results.append(run_one_seed(seed, cache, labeled, feature_cols, y, train_idx, val_idx, test_idx))

    aurocs = [r["fed_xgb_bagging_test_metrics"]["auroc"] for r in per_seed_results]

    results = {
        "step": 4,
        "description": "Federated GBDT via Flower's real FedXgbBagging aggregate() function (XGBoost tree-bagging, NOT FedTree histogram aggregation, NOT LightGBM) -- see module docstring for full honesty disclosure.",
        "algorithm_disclosure": {
            "library": "flwr 1.38.0, flwr.server.strategy.fedxgb_bagging.aggregate (real, unmodified library function)",
            "base_learner": "XGBoost 3.4.1 (NOT LightGBM)",
            "federation_mechanism": "tree-level bagging aggregation (whole trees concatenated across clients/rounds), NOT per-feature-bin gradient/Hessian histogram aggregation (FedTree/SecureBoost's actual mechanism)",
            "client_side_logic_source": "re-implemented faithfully from adap/flower examples/xgboost-comprehensive/xgboost_comprehensive/client_app.py:_local_boost (example/app code, not installed-library code) -- not invented from scratch",
            "fedtree_reverify_result": "pip install fedtree re-attempted fresh this session, still fails to build (libFedTree.dylib missing, needs CMake source build) -- identical to prior session's finding",
        },
        "n_clients": C.N_CLIENTS,
        "partition_seeds": C.PARTITION_SEEDS,
        "hyperparams": {"num_local_round_per_client_per_communication_round": NUM_LOCAL_ROUND, "n_rounds": N_ROUNDS, "xgb_params": XGB_BASE_PARAMS},
        "per_seed_results": per_seed_results,
        "partition_dependence": {
            "auroc_mean": float(np.mean(aurocs)),
            "auroc_std": float(np.std(aurocs)),
            "auroc_by_seed": dict(zip(C.PARTITION_SEEDS, aurocs)),
        },
        "runtime_seconds": time.time() - t0,
    }

    json_path = SCRIPT_DIR / "psyche_d_fedavg_vs_fedtree_step4_results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {json_path}")
    print(f"\nFedXgbBagging AUROC by seed: {results['partition_dependence']['auroc_by_seed']}")
    print(f"mean={results['partition_dependence']['auroc_mean']:.4f} std={results['partition_dependence']['auroc_std']:.4f}")
    return results


if __name__ == "__main__":
    run()
