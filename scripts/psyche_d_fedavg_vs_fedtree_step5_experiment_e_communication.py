#!/usr/bin/env python3
"""Experiment E (task Part 6.E): communication-efficiency estimate for
FedAvg MLP, FedXgbBagging, and TreeAvg -- model/update size per round,
total communication, rounds, and measured wall-clock time (from the actual
runs already executed for Experiments A-D; no separate timing run needed
since all three scripts' runtimes were already captured in their own
results JSONs).

TreeAvg's communication cost is UNCHANGED by the n_k/N weighting
correction: the transfer unit is still exactly 1 tree/client/round (the
weighting is a server-side arithmetic operation applied AFTER the same
bytes have already been transferred), so this script's numbers are
inherited directly from the corrected Step 5 results JSON with no
recomputation needed here.

FedXgbBagging's per-tree JSON size was NOT independently re-measured (it
uses the identical XGB_BASE_PARAMS/max_depth/dataset scale as TreeAvg, so
this script REUSES TreeAvg's measured mean bytes-per-tree as a size proxy
for FedXgbBagging's trees too -- disclosed explicitly here, not silently
assumed to be free)."""
from __future__ import annotations

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
N_CLIENTS = 5


def run() -> dict:
    step1 = json.loads((SCRIPT_DIR / "psyche_d_fedavg_vs_fedtree_step1_results.json").read_text())
    step23 = json.loads((SCRIPT_DIR / "psyche_d_fedavg_vs_fedtree_step23_results.json").read_text())
    step4 = json.loads((SCRIPT_DIR / "psyche_d_fedavg_vs_fedtree_step4_results.json").read_text())
    step5 = json.loads((SCRIPT_DIR / "psyche_d_fedavg_vs_fedtree_step5_results.json").read_text())

    # --- FedAvg MLP ---
    n_params = step1["architecture"]["n_params"]
    bytes_per_param = 4  # float32
    mlp_update_bytes = n_params * bytes_per_param
    fedavg_n_rounds = step23["per_seed_results"][0]["fedavg_n_rounds_run"]
    # per round: each of 5 clients uploads its full local state (n_params), server
    # broadcasts the new global state back to each client (download) -- 2*K transfers/round
    fedavg_total_bytes = mlp_update_bytes * N_CLIENTS * 2 * fedavg_n_rounds

    # --- FedXgbBagging ---
    fedxgb_n_rounds = step4["hyperparams"]["n_rounds"]
    fedxgb_trees_per_round = N_CLIENTS  # 1 tree/client/round
    # per-tree size REUSED from TreeAvg's measurement (same XGB params/depth/data scale)
    bytes_per_tree_proxy = sum(r["approx_bytes_per_tree_json"] for r in step5["per_seed_results"]) / len(step5["per_seed_results"])
    # per round: K clients upload 1 tree each; server broadcasts the K new trees back to
    # every client (so every client's next-round base model is up to date) -- 2*K tree-transfers/round
    fedxgb_total_bytes = bytes_per_tree_proxy * fedxgb_trees_per_round * 2 * fedxgb_n_rounds

    # --- TreeAvg (measured directly in Step 5, corrected n_k/N weighting) ---
    treeavg_n_rounds = step5["hyperparams"]["n_rounds"]
    treeavg_bytes_by_seed = [r["approx_total_communication_bytes"] for r in step5["per_seed_results"]]
    treeavg_total_bytes = sum(treeavg_bytes_by_seed) / len(treeavg_bytes_by_seed)

    results = {
        "experiment": "E -- communication efficiency",
        "note_fedxgb_tree_size": "FedXgbBagging's per-tree JSON size is REUSED from TreeAvg's own measurement (identical XGB hyperparams/max_depth/data scale), not independently re-measured -- disclosed, not silently assumed free.",
        "note_treeavg_weighting": "TreeAvg's n_k/N sample-count weighting (vs. the prior uniform 1/K) changes NOTHING about communication volume -- it is a server-side arithmetic operation on the same transferred bytes, not an additional transfer.",
        "fedavg_mlp": {
            "update_unit": "full model state_dict (float32)",
            "bytes_per_client_update": mlp_update_bytes,
            "n_clients": N_CLIENTS,
            "n_rounds": fedavg_n_rounds,
            "total_communication_bytes_upload_plus_download": fedavg_total_bytes,
            "total_communication_mb": fedavg_total_bytes / 1e6,
            "measured_wall_clock_seconds_all_3_seeds": step23["runtime_seconds"],
        },
        "fedxgb_bagging": {
            "update_unit": "1 new XGBoost tree (JSON) per client per round",
            "approx_bytes_per_tree": bytes_per_tree_proxy,
            "n_clients": N_CLIENTS,
            "n_rounds": fedxgb_n_rounds,
            "final_n_trees": N_CLIENTS * fedxgb_n_rounds,
            "total_communication_bytes_upload_plus_download": fedxgb_total_bytes,
            "total_communication_mb": fedxgb_total_bytes / 1e6,
            "measured_wall_clock_seconds_all_3_seeds": step4["runtime_seconds"],
        },
        "tree_avg": {
            "update_unit": "1 new XGBoost tree (JSON) per client per round -- SAME transfer unit as FedXgbBagging; the n_k/N-weighted averaging is a server-side arithmetic operation on already-transferred trees, not an extra communication cost",
            "approx_bytes_per_tree_by_seed": {str(r["partition_seed"]): r["approx_bytes_per_tree_json"] for r in step5["per_seed_results"]},
            "n_clients": N_CLIENTS,
            "n_rounds": treeavg_n_rounds,
            "final_n_trees_total": N_CLIENTS * treeavg_n_rounds,
            "total_communication_bytes_by_seed": {str(r["partition_seed"]): r["approx_total_communication_bytes"] for r in step5["per_seed_results"]},
            "total_communication_mb_mean": treeavg_total_bytes / 1e6,
            "measured_wall_clock_seconds_all_3_seeds": step5["runtime_seconds"],
        },
        "interpretation": (
            "TreeAvg and FedXgbBagging have IDENTICAL communication cost per round "
            "(same transfer unit: 1 tree/client/round, same 30 rounds, same 5 clients) "
            "-- TreeAvg's n_k/N-weighted averaging is purely a server-side computation "
            "on the SAME bytes already being transferred for bagging, not an additional "
            "cost. FedAvg MLP's communication is smaller in absolute count of bytes "
            "here only because this particular MLP (11905 params, ~46.5KB/update) "
            "happens to be compact relative to 150-tree XGBoost ensembles at this "
            "depth/eta; this is a property of THESE specific model sizes, not a "
            "general rule that neural nets communicate less than tree ensembles."
        ),
    }

    out_path = SCRIPT_DIR / "psyche_d_fedavg_vs_fedtree_step5_experiment_e_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {out_path}")
    print(f"FedAvg MLP: {results['fedavg_mlp']['total_communication_mb']:.2f} MB total, {fedavg_n_rounds} rounds, wall-clock {step23['runtime_seconds']:.1f}s")
    print(f"FedXgbBagging: {results['fedxgb_bagging']['total_communication_mb']:.2f} MB total, {fedxgb_n_rounds} rounds, wall-clock {step4['runtime_seconds']:.1f}s")
    print(f"TreeAvg: {results['tree_avg']['total_communication_mb_mean']:.2f} MB total, {treeavg_n_rounds} rounds, wall-clock {step5['runtime_seconds']:.1f}s")
    return results


if __name__ == "__main__":
    run()
