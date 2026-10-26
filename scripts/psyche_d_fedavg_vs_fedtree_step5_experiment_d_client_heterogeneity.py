#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. `research_datasets/` is
# gitignored; local research data, never committed to this repo.
"""Experiment D (task Part 6.D): characterize the 5 real (non-synthetic)
clients produced by `assign_clients()` at partition_seed=1 -- participants,
samples, positive rate, and baseline-PHQ-severity distribution differences
-- then relate this to the federated methods' measured behavior from
Experiments A/C. No artificial non-IID partition is constructed (per task
instruction) -- these are the SAME whole-participant, seeded-permutation
client splits already used throughout Steps 2-5.

Also reports each client's TreeAvg n_k/N weight directly (client row count
/ total row count at this seed) -- this IS the quantity the corrected
Step-5 script uses as its per-client combination weight, so this
experiment doubles as the client-size record behind that weighting
decision, not just a separate descriptive statistic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import psyche_d_fedavg_vs_fedtree_common as C  # noqa: E402
from psyche_d_fedavg_vs_fedtree_step4_fed_xgb_bagging import load_split_and_frame  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
SEED = 1


def run() -> dict:
    cache, labeled, feature_cols, y, train_idx, val_idx, test_idx = load_split_and_frame()
    train_df = labeled.loc[train_idx]
    sites = C.assign_clients(cache["train_participants"], k=C.N_CLIENTS, seed=SEED)
    client_dfs = [train_df[train_df["user_id"].isin(s)] for s in sites]

    n_rows = np.array([len(d) for d in client_dfs], dtype=np.float64)
    N_total = float(n_rows.sum())
    treeavg_weights = n_rows / N_total

    client_stats = []
    for i, d in enumerate(client_dfs):
        yc = y.loc[d.index].to_numpy()
        phq_start = d["phq9_score_start"].to_numpy(dtype=float)
        client_stats.append({
            "client_id": i,
            "n_participants": len(sites[i]),
            "n_rows": len(d),
            "treeavg_weight_n_k_over_N": float(treeavg_weights[i]),
            "positive_rate": float(yc.mean()),
            "n_positive": int(yc.sum()),
            "phq9_score_start_mean": float(np.nanmean(phq_start)),
            "phq9_score_start_std": float(np.nanstd(phq_start)),
        })

    positive_rates = [c["positive_rate"] for c in client_stats]
    phq_means = [c["phq9_score_start_mean"] for c in client_stats]

    results = {
        "experiment": "D -- client heterogeneity characterization",
        "partition_seed_used": SEED,
        "n_clients": C.N_CLIENTS,
        "N_total_rows": N_total,
        "client_stats": client_stats,
        "positive_rate_range": [min(positive_rates), max(positive_rates)],
        "positive_rate_std_across_clients": float(np.std(positive_rates)),
        "baseline_phq_mean_range": [min(phq_means), max(phq_means)],
        "baseline_phq_mean_std_across_clients": float(np.std(phq_means)),
        "treeavg_weight_range": [float(treeavg_weights.min()), float(treeavg_weights.max())],
        "uniform_weight_would_have_been": 1.0 / C.N_CLIENTS,
        "interpretation": (
            "Clients are whole-participant IID-ish random splits (not manufactured "
            "non-IID) -- positive-rate and baseline-PHQ-mean spread across clients "
            "is real but modest (see std above), consistent with this being a "
            "seeded-permutation random partition of the same underlying population, "
            "not a deliberately skewed split. The client row-count spread (see "
            "treeavg_weight_range above) is what makes TreeAvg's sample-count-"
            "weighted n_k/N combination rule numerically different from a uniform "
            "1/K average -- these weights are the actual per-client coefficients "
            "used by the corrected Step-5 script, not an illustrative aside. Any "
            "federated-vs-centralized performance gap measured in Experiments A/B "
            "is therefore attributable to the federation MECHANISM (periodic "
            "weighted averaging over 5 smaller local datasets) rather than to "
            "engineered distributional skew across clients."
        ),
    }

    out_path = SCRIPT_DIR / "psyche_d_fedavg_vs_fedtree_step5_experiment_d_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {out_path}")
    for c in client_stats:
        print(f"  client {c['client_id']}: n_participants={c['n_participants']} n_rows={c['n_rows']} w_k={c['treeavg_weight_n_k_over_N']:.4f} pos_rate={c['positive_rate']:.4f} phq_start_mean={c['phq9_score_start_mean']:.3f}")
    print(f"positive_rate range: {results['positive_rate_range']} std={results['positive_rate_std_across_clients']:.4f}")
    print(f"treeavg n_k/N weight range: {results['treeavg_weight_range']} (uniform 1/K would have been {results['uniform_weight_would_have_been']:.4f} for every client)")
    return results


if __name__ == "__main__":
    run()
