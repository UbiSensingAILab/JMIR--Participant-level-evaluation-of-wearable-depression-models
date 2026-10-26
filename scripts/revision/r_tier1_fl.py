#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. `research_datasets/` is gitignored.
"""Updated analysis plan, Tier 1, R8: federated learning with the validation-selected
TreeAvg learning rate (eta 0.05) as the primary result and eta 0.10 as sensitivity.

All predictions are the saved test-set predictions (row order = fixed split test rows,
checked against y_test): FedAvg step23, FedXgbBagging step4, TreeAvg eta 0.10 step5,
TreeAvg eta 0.05 s5 grid (selected by mean validation AUROC), centralized LightGBM step1
(frozen configuration, 70% train), centralized XGBoost s4. Participant bootstrap on the
606 test participants, 2000 draws, seed 42, same draws for every model and seed.
Pooled difference = mean of the 3 seed-aligned differences within each draw.
Local-only baselines (R8d) were run in scripts/revision/p2_fl_core.py (P2_3_*), reused here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import rev_common as RC  # noqa: E402
from psyche_d_fedavg_vs_fedtree_step4_fed_xgb_bagging import load_split_and_frame  # noqa: E402

S = RC.ROOT / "scripts"
J = lambda f: json.loads((S / f).read_text())  # noqa: E731
cache, labeled, fc, yser, tr, va, te = load_split_and_frame()
y = yser.loc[te].to_numpy()
g = labeled.loc[te, "user_id"].to_numpy()
draws = RC.participant_draws(g)


def seeds(f, key):
    d = J(f)
    out = {}
    for r in d["per_seed_results"]:
        assert np.array_equal(np.asarray(r["y_test"]), y)
        out[r["partition_seed"]] = np.asarray(r[key])
    return out


preds = {"FedAvg": seeds("psyche_d_fedavg_vs_fedtree_step23_results.json", "fedavg_test_predictions"),
         "FedXgbBagging": seeds("psyche_d_fedavg_vs_fedtree_step4_results.json", "fed_xgb_bagging_test_predictions"),
         "TreeAvg eta 0.10 (sensitivity)": seeds("psyche_d_fedavg_vs_fedtree_step5_results.json", "tree_avg_test_predictions")}
grid = J("psyche_d_treeavg_audit_extend_s5_results.json")
assert grid["selected_eta"] == 0.05
preds["TreeAvg eta 0.05 (primary)"] = {int(k): np.asarray(v["test_predictions"]) for k, v in grid["grid_results_by_eta_then_seed"]["0.05"].items()}
c1, c4 = J("psyche_d_fedavg_vs_fedtree_step1_results.json"), J("psyche_d_treeavg_audit_extend_s4_results.json")
assert np.array_equal(np.asarray(c1["y_test"]), y) and np.array_equal(np.asarray(c4["y_test"]), y)
central = {"Centralized LightGBM": np.asarray(c1["lightgbm_dart_same_split_test_predictions"]),
           "Centralized XGBoost": np.asarray(c4["centralized_xgboost_test_predictions"])}

M = {}
rows = []
for m, by_seed in preds.items():
    for s_, p in sorted(by_seed.items()):
        M[(m, s_)] = RC.summarize(y, p, draws)
        rows.append({"method": m, "partition_seed": s_, **RC.strip(M[(m, s_)])})
for m, p in central.items():
    M[(m, 0)] = RC.summarize(y, p, draws)
    rows.append({"method": m, "partition_seed": "n/a", **RC.strip(M[(m, 0)])})
per_seed = pd.DataFrame(rows)
summ = per_seed[per_seed["partition_seed"] != "n/a"].groupby("method")["auroc"].agg(mean="mean", sd_ddof1=lambda v: v.std(ddof=1)).reset_index()

PR = []
prim = "TreeAvg eta 0.05 (primary)"
for comp in ["FedAvg", "FedXgbBagging", "Centralized LightGBM", "Centralized XGBoost", "TreeAvg eta 0.10 (sensitivity)"]:
    diffs = []
    for s_ in (1, 2, 3):
        b = M[(comp, 0)] if comp.startswith("Centralized") else M[(comp, s_)]
        r = RC.paired(M[(prim, s_)], b)
        diffs.append((M[(prim, s_)]["_boot_auroc"] - b["_boot_auroc"], r["delta"]))
        PR.append({"comparison": f"TreeAvg(0.05) - {comp}", "partition_seed": s_, **r})
    boot = np.mean([d[0] for d in diffs], axis=0)
    lo, hi = RC.ci(boot)
    PR.append({"comparison": f"TreeAvg(0.05) - {comp}", "partition_seed": "pooled", "delta": float(np.mean([d[1] for d in diffs])),
               "ci": [lo, hi], "ci_excludes_0": bool(lo > 0 or hi < 0)})
local = pd.read_csv(RC.OUT / "P2_3_local_only_summary.csv")

per_seed.to_csv(RC.OUT / "R8_per_seed_auroc.csv", index=False)
summ.to_csv(RC.OUT / "R8_seed_mean_sd.csv", index=False)
pd.DataFrame(PR).to_csv(RC.OUT / "R8_paired_differences.csv", index=False)
res = {"seeds": {"split": 20260928, "partition": [1, 2, 3], "bootstrap": RC.BOOT_SEED, "n_boot": RC.N_BOOT},
       "treeavg_eta_selection": {"grid_mean_val_auroc": grid["mean_val_auroc_by_eta"], "selected": 0.05,
                                 "best_rounds_eta_0.05": {k: v["best_round"] for k, v in grid["grid_results_by_eta_then_seed"]["0.05"].items()}},
       "per_seed": rows, "mean_sd": summ.to_dict("records"), "paired": PR,
       "local_only_source": "results/revision/P2_3_local_only_summary.csv", "local_only": local.to_dict("records")}
(RC.OUT / "R8_results.json").write_text(json.dumps(res, indent=1, default=float))
print(per_seed[["method", "partition_seed", "auroc", "auroc_ci"]].to_string())
print(summ.to_string())
for r in PR:
    print(f"{r['comparison']:50s} {str(r['partition_seed']):7s} {r['delta']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]")
