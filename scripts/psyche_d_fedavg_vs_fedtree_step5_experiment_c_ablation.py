#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. `research_datasets/` is
# gitignored; local research data, never committed to this repo.
"""Experiment C (task Part 6.C, feature-source ablation): baseline-PHQ-only
vs. wearable/context-only vs. combined, for the federated tree methods
(TreeAvg and FedXgbBagging), reusing the ALREADY-ESTABLISHED feature-group
definitions from `scripts/ablation_psyche_d_severity_change_feature_sources.py`
(BASELINE_PHQ_COLS = ["phq9_score_start", "phq9_cat_start"]; wearable/context
= the other 150 columns) rather than re-inventing a new split.

This calls `run_one_seed` directly from the CORRECTED
`psyche_d_fedavg_vs_fedtree_step5_tree_avg` module -- i.e. it automatically
exercises the sample-count-weighted (n_k/N) TreeAvg combination rule (the
function is imported, not reimplemented, so there is no separate weighting
logic here to get wrong).

SCOPE REDUCTION (disclosed): run at partition_seed=1 ONLY (not all 3
seeds) to keep this ablation's compute bounded -- the purpose here is to
quantify how much signal comes from baseline PHQ vs. wearable/context, not
to re-establish partition-seed variance (already characterized in
Experiment A for the full feature set).

Baseline PHQ is available at prediction time and is not leakage (it is
the participant's own SCORE AT THE START of the quarter being predicted
over, already disclosed and used this way throughout this project's prior
PSYCHE-D work -- see PD18/PD19 in `train_psyche_d_severity_change.py` and
this session's own Part 2 instructions)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import psyche_d_fedavg_vs_fedtree_common as C  # noqa: E402
from psyche_d_fedavg_vs_fedtree_step4_fed_xgb_bagging import (  # noqa: E402
    load_split_and_frame as load4,
)
from psyche_d_fedavg_vs_fedtree_step4_fed_xgb_bagging import (  # noqa: E402
    run_one_seed as run_fedxgb_one_seed,
)
from psyche_d_fedavg_vs_fedtree_step5_tree_avg import (  # noqa: E402
    run_one_seed as run_treeavg_one_seed,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SEED = 1

BASELINE_PHQ_COLS = list(C.BASELINE_SEVERITY_COLS)  # ["phq9_score_start", "phq9_cat_start"]


def run() -> dict:
    cache, labeled, feature_cols_full, y, train_idx, val_idx, test_idx = load4()
    wearable_only_cols = [c for c in feature_cols_full if c not in BASELINE_PHQ_COLS]
    assert len(wearable_only_cols) == 150
    assert len(BASELINE_PHQ_COLS) == 2
    assert len(feature_cols_full) == 152

    feature_groups = {
        "baseline_phq_only": BASELINE_PHQ_COLS,
        "wearable_context_only": wearable_only_cols,
        "wearable_context_plus_baseline_phq": feature_cols_full,
    }

    results = {
        "experiment": "C -- feature-source ablation",
        "partition_seed_used": SEED,
        "note": "single-seed ablation, scope-reduced per module docstring; TreeAvg arm uses the corrected sample-count-weighted (n_k/N) combination rule via direct import of step5_tree_avg.run_one_seed",
        "arms": {},
    }
    for group_name, cols in feature_groups.items():
        print(f"\n=== TreeAvg / {group_name} ({len(cols)} features) ===")
        tree_avg_res = run_treeavg_one_seed(SEED, cache, labeled, cols, y, train_idx, val_idx, test_idx)
        print(f"\n=== FedXgbBagging / {group_name} ({len(cols)} features) ===")
        fedxgb_res = run_fedxgb_one_seed(SEED, cache, labeled, cols, y, train_idx, val_idx, test_idx)
        results["arms"][group_name] = {
            "n_features": len(cols),
            "tree_avg": {
                "auroc": tree_avg_res["tree_avg_test_metrics"]["auroc"],
                "auprc": tree_avg_res["tree_avg_test_metrics"]["auprc"],
                "client_weights_n_k_over_N": tree_avg_res["client_weights_n_k_over_N"],
                "full_metrics": tree_avg_res["tree_avg_test_metrics"],
            },
            "fedxgb_bagging": {"auroc": fedxgb_res["fed_xgb_bagging_test_metrics"]["auroc"], "auprc": fedxgb_res["fed_xgb_bagging_test_metrics"]["auprc"], "full_metrics": fedxgb_res["fed_xgb_bagging_test_metrics"]},
        }

    out_path = SCRIPT_DIR / "psyche_d_fedavg_vs_fedtree_step5_experiment_c_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")
    print("\n=== Summary (AUROC) ===")
    for group_name, arm in results["arms"].items():
        print(f"  {group_name:38s} TreeAvg={arm['tree_avg']['auroc']:.4f}  FedXgbBagging={arm['fedxgb_bagging']['auroc']:.4f}")
    return results


if __name__ == "__main__":
    run()
