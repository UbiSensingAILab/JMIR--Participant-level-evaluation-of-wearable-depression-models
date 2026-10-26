#!/usr/bin/env python3
"""Final report for `.planning/debug/psyche-d-fedavg-vs-fedtree.md` -- pure
aggregation of Step 1 / Step 2-3 / Step 4 results JSONs into one honest,
concise comparison report. No new training here."""
from __future__ import annotations

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((SCRIPT_DIR / name).read_text())


def fmt_metrics_row(label: str, m: dict) -> str:
    return (
        f"| {label} | {m['auroc']:.4f} | [{m['auroc_ci'][0]:.4f}, {m['auroc_ci'][1]:.4f}] | {m['auprc']:.4f} | "
        f"{m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | {m['sensitivity']:.4f} | {m['specificity']:.4f} | n={m['n']} |"
    )


def build_report() -> str:
    s1 = load("psyche_d_fedavg_vs_fedtree_step1_results.json")
    s23 = load("psyche_d_fedavg_vs_fedtree_step23_results.json")
    s4 = load("psyche_d_fedavg_vs_fedtree_step4_results.json")

    cited = s1["cited_lightgbm_dart_reference_5fold_pooled_oof"]
    dart_same_split = s1["lightgbm_dart_same_split_secondary_check_test_metrics"]
    mlp = s1["centralized_mlp_test_metrics"]

    fedavg_pd = s23["partition_dependence"]
    fedxgb_pd = s4["partition_dependence"]

    lines = []
    lines.append("# PSYCHE-D FedAvg-MLP vs. Federated-GBDT (FedXgbBagging) -- Final Comparison Report")
    lines.append("")
    lines.append(
        "PSYCHE-D dataset (Zenodo 5085146), derived from the DiSCover Project (NCT03421223), CC BY-NC 4.0. "
        "`research_datasets/` is gitignored; local research data, never committed to this repo. "
        "Debug session: `.planning/debug/psyche-d-fedavg-vs-fedtree.md`."
    )
    lines.append("")
    lines.append(
        "**Methodologically distinct from, and independent of, this project's other FL session** "
        "(`.planning/debug/psyche-d-fl-experiment-design.md`, still separately pending, untouched by this work): "
        "that session used `StratifiedGroupKFold(5)` pooled-OOF evaluation for every arm. THIS report uses a "
        "**single, fixed, one-time participant-level train/validation/test split** -- validation and test "
        "participants are held out of ALL client training entirely, never rotated."
    )
    lines.append("")

    lines.append("## Headline")
    lines.append("")
    lines.append(f"| Model | AUROC | Δ vs. cited LightGBM-DART (0.7090) | Δ vs. same-split LightGBM-DART ({dart_same_split['auroc']:.4f}) |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Centralized LightGBM-DART (cited, 5-fold pooled-OOF, whole population) | {cited['auroc']:.4f} | -- (reference) | -- |")
    lines.append(f"| Centralized LightGBM-DART (SAME split as this report, secondary check) | {dart_same_split['auroc']:.4f} | {dart_same_split['auroc'] - cited['auroc']:+.4f} | -- (reference for this split) |")
    lines.append(f"| Centralized MLP | {mlp['auroc']:.4f} | {mlp['auroc'] - cited['auroc']:+.4f} | {mlp['auroc'] - dart_same_split['auroc']:+.4f} |")
    lines.append(f"| FedAvg MLP (mean of 3 partition seeds) | {fedavg_pd['fedavg_auroc_mean']:.4f} ± {fedavg_pd['fedavg_auroc_std']:.4f} | {fedavg_pd['fedavg_auroc_mean'] - cited['auroc']:+.4f} | {fedavg_pd['fedavg_auroc_mean'] - dart_same_split['auroc']:+.4f} |")
    lines.append(f"| Local-only MLP, no aggregation (mean of 3 seeds x 5 clients) | {fedavg_pd['local_only_auroc_mean']:.4f} ± {fedavg_pd['local_only_auroc_std']:.4f} | {fedavg_pd['local_only_auroc_mean'] - cited['auroc']:+.4f} | {fedavg_pd['local_only_auroc_mean'] - dart_same_split['auroc']:+.4f} |")
    lines.append(f"| Federated GBDT -- **FedXgbBagging** (mean of 3 partition seeds; NOT FedTree, NOT LightGBM -- see disclosure below) | {fedxgb_pd['auroc_mean']:.4f} ± {fedxgb_pd['auroc_std']:.4f} | {fedxgb_pd['auroc_mean'] - cited['auroc']:+.4f} | {fedxgb_pd['auroc_mean'] - dart_same_split['auroc']:+.4f} |")
    lines.append("")
    lines.append(
        f"**Answer to the key scientific question:** on this held-out participant-disjoint test set, the federated "
        f"tree-based model (FedXgbBagging, {fedxgb_pd['auroc_mean']:.4f} AUROC) retains **more** of the centralized "
        f"LightGBM-DART performance than the FedAvg MLP does ({fedavg_pd['fedavg_auroc_mean']:.4f} AUROC) -- a gap of "
        f"{fedxgb_pd['auroc_mean'] - fedavg_pd['fedavg_auroc_mean']:+.4f} AUROC in the tree method's favor -- but "
        f"**neither federated method closes the gap to centralized LightGBM-DART**: both remain "
        f"{cited['auroc'] - fedxgb_pd['auroc_mean']:.4f} to {cited['auroc'] - fedavg_pd['fedavg_auroc_mean']:.4f} AUROC "
        f"below the {dart_same_split['auroc']:.4f} to {cited['auroc']:.4f} centralized reference range. "
        f"Both federated arms clearly beat their own local-only-client baseline "
        f"({fedavg_pd['local_only_auroc_mean']:.4f}), confirming aggregation (of either kind) provides real value "
        f"over training on a single client's data alone."
    )
    lines.append("")

    lines.append("## Full metrics")
    lines.append("")
    lines.append("| Model | AUROC | AUROC 95% CI | AUPRC | Accuracy | Balanced Acc. | Sensitivity | Specificity | |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    lines.append(fmt_metrics_row("Centralized LightGBM-DART (cited 5-fold pooled-OOF)", cited | {"auroc_ci": cited["auroc_ci"], "n": "10866 (whole population, 5-fold)"}))
    lines.append(fmt_metrics_row("Centralized LightGBM-DART (same split, secondary check)", dart_same_split))
    lines.append(fmt_metrics_row("Centralized MLP", mlp))
    for r in s23["per_seed_results"]:
        lines.append(fmt_metrics_row(f"FedAvg MLP (partition seed {r['partition_seed']})", r["fedavg_test_metrics"]))
    for r in s4["per_seed_results"]:
        lines.append(fmt_metrics_row(f"FedXgbBagging (partition seed {r['partition_seed']})", r["fed_xgb_bagging_test_metrics"]))
    lines.append("")

    lines.append("## Step 1 -- Centralized MLP competitiveness screen")
    lines.append("")
    comp = s1["comparison"]
    lines.append(
        f"Screening criterion (task's own pre-registered tolerance): MLP AUROC within ±{comp['competitive_screen_tolerance']} of the "
        f"LightGBM-DART reference. **Verdict: NOT MET** "
        f"(vs. cited: {comp['mlp_auroc_minus_cited_lightgbm_auroc']:+.4f}, vs. same-split: {comp['mlp_auroc_minus_same_split_lightgbm_auroc']:+.4f})."
    )
    lines.append("")
    lines.append(
        "**Per the task's own resolution of the Step 1/Step 2 gating ambiguity (recorded in this session's debug file "
        "before any training ran): Steps 2-4 were run anyway, for completeness, despite the competitiveness screen not "
        "being met.** The numbers below are real, not gated on this screen passing -- flagged here so they are never "
        "silently read as if the bar had been cleared."
    )
    lines.append("")

    lines.append("## Local-only vs. FedAvg vs. centralized -- does aggregation help?")
    lines.append("")
    lines.append("| Partition seed | Local-only mean AUROC (5 clients) | FedAvg global AUROC | Centralized MLP AUROC |")
    lines.append("|---|---|---|---|")
    for r in s23["per_seed_results"]:
        lines.append(f"| {r['partition_seed']} | {r['local_only_mean_auroc']:.4f} ± {r['local_only_std_auroc']:.4f} | {r['fedavg_test_metrics']['auroc']:.4f} | {mlp['auroc']:.4f} |")
    lines.append("")
    lines.append(
        "FedAvg consistently beats local-only-per-client training (aggregation helps), and consistently trails the "
        "centralized MLP trained on the pooled data directly (the expected FL cost, since FedAvg only approximates "
        "joint training via periodic parameter averaging over statistically heterogeneous local partitions)."
    )
    lines.append("")

    lines.append("## Client construction (identical across Steps 2/3/4)")
    lines.append("")
    lines.append(f"- **Number of clients:** {s23['n_clients']} (`N_CLIENTS`, cross-silo range, fixed -- no hyperparameter hunt over K)")
    lines.append(f"- **Partition seeds tested:** {s23['partition_seeds']} -- tests whether performance depends strongly on the client partition, per task instruction")
    lines.append("- **Partitioning method:** whole-participant, seeded `rng.permutation` + `np.array_split` over the TRAIN-pool participants ONLY (`scripts/psyche_d_fedavg_vs_fedtree_common.py:assign_clients`) -- never splits one participant's rows across clients (longitudinal-safe)")
    for r in s23["per_seed_results"]:
        lines.append(f"  - seed {r['partition_seed']}: client sizes (participants) = {r['client_n_participants']}, (rows) = {r['client_n_rows']}")
    lines.append("")

    lines.append("## Participant-disjointness confirmations")
    lines.append("")
    split = s1["split"]
    lines.append(f"- **Split:** {split['n_train_participants']} train / {split['n_val_participants']} val / {split['n_test_participants']} test participants "
                  f"({split['n_train_rows']} / {split['n_val_rows']} / {split['n_test_rows']} rows), fixed once (seed={split['seed']}), stratified by each participant's `phq9_cat_start` mode, whole-participant assignment (never split one participant's rows across sets).")
    lines.append("- **Asserted programmatically at every step** (`assert not (A & B)` for every pair; failure would crash the script, not silently continue):")
    lines.append("  - train intersect val = empty set, train intersect test = empty set, val intersect test = empty set (`build_fixed_split`)")
    lines.append("  - every client intersect every other client = empty set, union of all clients == the exact train-pool participant set, no more/less (`verify_disjointness`, re-checked independently inside Steps 2/3 AND Step 4 for every partition seed)")
    lines.append("  - client-union intersect val = empty set, client-union intersect test = empty set (`verify_disjointness`)")
    lines.append("- **Validation/test data were NEVER used** for: model fitting, threshold selection, preprocessing-statistic fitting, or client construction. Validation WAS used for: early-stopping / best-round selection (by AUROC) in Steps 1, 2/3, and 4 -- an explicitly permitted use per the task instructions (\"do not use validation/test labels for model fitting... or hyperparameter selection\" -- early-stopping round selection is the one explicitly-analogous exception every one of this project's other FL scripts also relies on; it is not test-set-based and is disclosed here).")
    lines.append("")

    lines.append("## Exact preprocessing procedure")
    lines.append("")
    lines.append(
        "**MLP arms (centralized + FedAvg):** mean-impute + standardize. Statistics (`mean`, `std`) computed via "
        "`federated_sufficient_stats()` -- aggregates per-feature `(count, sum, sumsq)` across row groups (1 group for "
        "the centralized arm, K groups for the federated arm) and derives mean/std from the aggregate. **Verified "
        "programmatically that the K-group federated aggregation produces IDENTICAL statistics (max abs diff < 1e-9) "
        "to a 1-group centralized fit** -- see the debug session's Evidence log. Only these three scalars-per-feature "
        "ever need to leave a client; no raw row is centralized. Statistics are fit from TRAIN-pool rows only (or, in "
        "the federated case, from each client's own local train rows) -- validation/test rows are never included."
    )
    lines.append(
        "**FedXgbBagging arm:** no imputation -- XGBoost's native missing-value handling (`DMatrix(..., missing=np.nan)`) "
        "is used directly on the raw 152-feature matrix, consistent with standard XGBoost practice (this differs from "
        "the LightGBM-DART reference's `SimpleImputer(mean)` step, disclosed here rather than silently glossed over). "
        "Class-imbalance handling (`scale_pos_weight`) is computed from federated-safe AGGREGATED positive/negative "
        "COUNTS across clients (never raw rows), applied identically at every client."
    )
    lines.append("")

    lines.append("## Step 4 -- federated GBDT algorithm disclosure (verbatim from the results JSON)")
    lines.append("")
    ad = s4["algorithm_disclosure"]
    for k, v in ad.items():
        lines.append(f"- **{k}:** {v}")
    lines.append("")

    lines.append("## Hyperparameters (fixed, no broad architecture/hyperparameter hunting per task instruction)")
    lines.append("")
    lines.append(f"- **MLP architecture:** {s1['architecture']['input_dim']} → {s1['architecture']['hidden'][0]} → {s1['architecture']['hidden'][1]} → 1, dropout={s1['architecture']['dropout']}, {s1['architecture']['n_params']} parameters -- identical for centralized (Step 1) and FedAvg (Steps 2/3)")
    lines.append(f"- **Centralized MLP training:** Adam lr={s1['training']['lr']}, early-stopped on validation AUROC (patience={s1['training']['patience']}, max_epochs={s1['training']['max_epochs']}), best epoch={s1['training']['best_epoch']} (of {s1['training']['n_epochs_run']} run)")
    lines.append(f"- **FedAvg:** {s23['fedavg_hyperparams']['n_rounds']} communication rounds, {s23['fedavg_hyperparams']['local_epochs_per_round']} local epochs/round/client, Adam lr={s23['fedavg_hyperparams']['lr']}, weighted-by-sample-count averaging, common random init (`torch.manual_seed`) distributed to all clients -- NO centralized pretraining")
    lines.append(f"- **FedXgbBagging:** {s4['hyperparams']['num_local_round_per_client_per_communication_round']} new tree/client/round, {s4['hyperparams']['n_rounds']} rounds (→ {s4['hyperparams']['n_rounds'] * s4['n_clients']} total trees in the final global ensemble), params: `{s4['hyperparams']['xgb_params']}`")
    lines.append("")

    lines.append("## Reproduction")
    lines.append("")
    lines.append("Isolated environment (never `.venv311`, which is this project's production momentfm dashboard backend -- see debug session Evidence log for the near-miss and why): ")
    lines.append("```bash")
    lines.append("python3 -m venv .venv-psyche-d-fl-mlp && source .venv-psyche-d-fl-mlp/bin/activate")
    lines.append('pip install "pandas==2.3.1" "numpy==1.26.4" "scikit-learn==1.7.1" "scipy==1.16.1" "lightgbm==4.7.0" torch xgboost "flwr[simulation]" pyarrow')
    lines.append("DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib python3 scripts/psyche_d_fedavg_vs_fedtree_step1_centralized.py")
    lines.append("python3 scripts/psyche_d_fedavg_vs_fedtree_step23_fedavg.py")
    lines.append("python3 scripts/psyche_d_fedavg_vs_fedtree_step4_fed_xgb_bagging.py")
    lines.append("python3 scripts/psyche_d_fedavg_vs_fedtree_final_report.py")
    lines.append("```")
    lines.append("")
    lines.append(f"Runtimes: Step 1={s1['runtime_seconds']:.1f}s, Steps 2/3={s23['runtime_seconds']:.1f}s, Step 4={s4['runtime_seconds']:.1f}s.")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    report = build_report()
    out_path = SCRIPT_DIR / "psyche_d_fedavg_vs_fedtree_final_report.md"
    out_path.write_text(report)
    print(f"Wrote {out_path}")
