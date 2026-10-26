#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. `research_datasets/` is
# gitignored; local research data, never committed to this repo.
"""Section 5 of .planning/debug/psyche-d-treeavg-audit-extend.md: a SMALL
controlled TreeAvg learning-rate study (not a huge sweep), run AFTER Section
3's validation-curve diagnostic (which found no evidence that more rounds
would help -- so this section isolates the learning-rate question alone,
keeping the round budget at the current 30-round anchor).

Grid: eta in {0.03, 0.05, 0.10 (current anchor), 0.20, 0.30}, x 3 partition
seeds = 15 runs. For each (eta, seed): train on training clients, select the
best round via VALIDATION AUROC only (never test), report test metrics at
that selected round purely for later reporting (never used to pick eta).

Implementation note: reuses the REAL, audited, unmodified
`psyche_d_fedavg_vs_fedtree_step5_tree_avg.run_one_seed` function for every
run -- eta is varied by monkeypatching the module's imported `XGB_BASE_PARAMS`
constant for the duration of each call only, then restoring it. This means
the n_k/N weighting, the client-side mechanic, the server-side combination,
and the best-round selection logic are ALL the exact same code already
audited in Section 2 -- nothing about the algorithm itself is reimplemented
or at risk of drifting from the audited version. N_ROUNDS is left at its
default (30, matching the current anchor and FedXgbBagging's budget) for
every eta in the grid -- this section is about learning rate, NOT round
count (that is Section 6, already resolved by Section 3's evidence).

Model selection: the eta with the highest MEAN validation AUROC across all
3 partition seeds is selected as "TreeAvg selected/final configuration" for
Section 17's final table -- chosen via validation data only, never test.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import psyche_d_fedavg_vs_fedtree_common as C  # noqa: E402
import psyche_d_fedavg_vs_fedtree_step5_tree_avg as S5  # noqa: E402
from psyche_d_fedavg_vs_fedtree_step4_fed_xgb_bagging import (  # noqa: E402
    XGB_BASE_PARAMS,
    load_split_and_frame,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ETA_GRID = [0.03, 0.05, 0.10, 0.20, 0.30]
CURRENT_ANCHOR_ETA = 0.10


def run_with_eta(eta: float, seed: int, cache, labeled, feature_cols, y, train_idx, val_idx, test_idx) -> dict:
    original_params = S5.XGB_BASE_PARAMS
    S5.XGB_BASE_PARAMS = dict(XGB_BASE_PARAMS, eta=eta)
    try:
        return S5.run_one_seed(seed, cache, labeled, feature_cols, y, train_idx, val_idx, test_idx)
    finally:
        S5.XGB_BASE_PARAMS = original_params  # always restore -- never leaves the module mutated


def run() -> dict:
    t0 = time.time()
    cache, labeled, feature_cols, y, train_idx, val_idx, test_idx = load_split_and_frame()

    grid_results = {}
    for eta in ETA_GRID:
        grid_results[eta] = {}
        for seed in C.PARTITION_SEEDS:
            print(f"\n=== TreeAvg eta={eta} partition_seed={seed} ===")
            r = run_with_eta(eta, seed, cache, labeled, feature_cols, y, train_idx, val_idx, test_idx)
            grid_results[eta][seed] = {
                "best_round": r["best_round"],
                "best_val_auroc": r["best_val_auroc"],
                "test_auroc": r["tree_avg_test_metrics"]["auroc"],
                "test_auprc": r["tree_avg_test_metrics"]["auprc"],
                "test_predictions": r["tree_avg_test_predictions"],
                "val_predictions": r["tree_avg_val_predictions"],
            }

    mean_val_auroc_by_eta = {
        eta: float(np.mean([grid_results[eta][s]["best_val_auroc"] for s in C.PARTITION_SEEDS]))
        for eta in ETA_GRID
    }
    mean_test_auroc_by_eta = {
        eta: float(np.mean([grid_results[eta][s]["test_auroc"] for s in C.PARTITION_SEEDS]))
        for eta in ETA_GRID
    }
    std_test_auroc_by_eta = {
        eta: float(np.std([grid_results[eta][s]["test_auroc"] for s in C.PARTITION_SEEDS]))
        for eta in ETA_GRID
    }

    selected_eta = max(mean_val_auroc_by_eta, key=mean_val_auroc_by_eta.get)
    selection_is_anchor = abs(selected_eta - CURRENT_ANCHOR_ETA) < 1e-9

    # Final reported test metrics for the SELECTED configuration (mean +/- SD
    # across 3 seeds, same convention as the official step5 results).
    selected_test_aurocs = [grid_results[selected_eta][s]["test_auroc"] for s in C.PARTITION_SEEDS]
    selected_test_auprcs = [grid_results[selected_eta][s]["test_auprc"] for s in C.PARTITION_SEEDS]

    results = {
        "section": 5,
        "description": "Small controlled TreeAvg learning-rate study -- round budget fixed at the current 30-round anchor (Section 3 found no evidence more rounds would help); only eta is varied. Selection via validation AUROC only, never test.",
        "eta_grid": ETA_GRID,
        "current_anchor_eta": CURRENT_ANCHOR_ETA,
        "grid_results_by_eta_then_seed": grid_results,
        "mean_val_auroc_by_eta": mean_val_auroc_by_eta,
        "mean_test_auroc_by_eta_FOR_REPORTING_ONLY_NOT_USED_FOR_SELECTION": mean_test_auroc_by_eta,
        "std_test_auroc_by_eta_FOR_REPORTING_ONLY": std_test_auroc_by_eta,
        "selected_eta": selected_eta,
        "selection_matches_current_anchor": selection_is_anchor,
        "selected_configuration_test_auroc_mean": float(np.mean(selected_test_aurocs)),
        "selected_configuration_test_auroc_std": float(np.std(selected_test_aurocs)),
        "selected_configuration_test_auprc_mean": float(np.mean(selected_test_auprcs)),
        "selected_configuration_test_auprc_std": float(np.std(selected_test_auprcs)),
        "conclusion": (
            f"Selected eta={selected_eta} by mean validation AUROC across 3 seeds "
            f"({'matches' if selection_is_anchor else 'DIFFERS FROM'} the current anchor eta={CURRENT_ANCHOR_ETA}). "
            "Mean test AUROC across the grid is reported for transparency but was NOT used to "
            "choose eta (test-set-blind selection, per task's explicit requirement)."
        ),
        "runtime_seconds": time.time() - t0,
    }

    out_path = SCRIPT_DIR / "psyche_d_treeavg_audit_extend_s5_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")
    print("\n=== Mean validation AUROC by eta (selection criterion) ===")
    for eta in ETA_GRID:
        print(f"  eta={eta}: mean_val_auroc={mean_val_auroc_by_eta[eta]:.4f}  mean_test_auroc={mean_test_auroc_by_eta[eta]:.4f} (+/-{std_test_auroc_by_eta[eta]:.4f})")
    print(f"\nSelected eta: {selected_eta} (matches anchor: {selection_is_anchor})")
    print(f"Selected config test AUROC: {results['selected_configuration_test_auroc_mean']:.4f} +/- {results['selected_configuration_test_auroc_std']:.4f}")
    return results


if __name__ == "__main__":
    run()
