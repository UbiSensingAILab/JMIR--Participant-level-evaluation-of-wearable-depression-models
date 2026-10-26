#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), derived from the DiSCover Project
# (NCT03421223), CC BY-NC 4.0, a separate institution's protocol,
# independent of this project's own ethics protocol (REB25-1587_MOD1).
"""MOMENT-Small features under the project's federated strategies.

Supplementary experiment, NOT part of the frozen primary results. The only
change versus the primary runs is the input: each sample's 152 features are
extended with its 512-d frozen MOMENT-Small embedding (from
psyche_d_moment_small.py, cached in research_datasets/, gitignored), giving
664 features. Everything else is reused unchanged by calling the existing
per-seed functions:

  FedAvg MLP      psyche_d_fedavg_vs_fedtree_step23_fedavg.run_one_seed
  FedXgbBagging   psyche_d_fedavg_vs_fedtree_step4_fed_xgb_bagging.run_one_seed
  TreeAvg         psyche_d_fedavg_vs_fedtree_step5_tree_avg.run_one_seed
                  (frozen anchor: n_k/N, eta 0.10, 30 rounds, 1 tree/client/round)
  Centralized XGBoost reference: same params/early stopping as audit-extend s4.

Same fixed split, same 3 partition seeds, same client assignments, same
validation-only early stopping. The MLP's input width is fixed at 152 in
psyche_d_fedavg_vs_fedtree_mlp_model.py; it is widened to 664 for this
process only (SmallMLP's default input_dim), the shared file is not edited.

Comparisons: each method with vs without MOMENT on the same seed, AUROC
difference with a participant-level paired bootstrap (2,000 resamples,
partition seed 1, as in s10), plus the validation-selected operating point
(argmax validation balanced accuracy, S13 rule). Aggregate metrics only.

Run (in .venv-psyche-d-fl-mlp), each part in its own process:
  OMP_NUM_THREADS=1 python scripts/psyche_d_moment_small_fl.py mlp
  python scripts/psyche_d_moment_small_fl.py trees
  python scripts/psyche_d_moment_small_fl.py report
OMP_NUM_THREADS=1 is required for the mlp part: torch training segfaults on
macOS when lightgbm (imported via the shared data loader) is in the same
process. Single-threaded FedAvg still reproduces the saved predictions
exactly (parity check in the report).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
import psyche_d_fedavg_vs_fedtree_common as C  # noqa: E402
import psyche_d_treeavg_audit_extend_s13_threshold_selection as S13  # noqa: E402
# torch (FedAvg) and xgboost (tree methods) each load their own OpenMP runtime;
# in one macOS process that segfaults, so each part runs in its own process
# and imports only what it needs. Intermediate predictions go to the
# gitignored research cache.

EMB_PATH = ROOT / "research_datasets" / "old" / "psyche_d" / "derived" / "moment_small_embeddings.npz"
OUT = SCRIPT_DIR / "psyche_d_moment_small_fl_results.json"
UI = ROOT / "src" / "data" / "psycheDMomentSmallFL.json"
C_SPLIT_CACHE = SCRIPT_DIR / "psyche_d_fedavg_vs_fedtree_split_cache.pkl"
PART_PATH = EMB_PATH.parent / "moment_small_fl_{}.npz"
N_BOOT = 2000


def op(y, p, t):
    pr = p >= t
    tp = int((pr & (y == 1)).sum()); fp = int((pr & (y == 0)).sum()); tn = int((~pr & (y == 0)).sum()); fn = int((~pr & (y == 1)).sum())
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    return {"threshold": float(t), "sensitivity": sens, "specificity": spec, "balanced_accuracy": (sens + spec) / 2,
            "ppv": tp / (tp + fp) if tp + fp else float("nan"), "accuracy": (tp + tn) / len(y), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def paired_ci(y, groups, a, b, seed=42):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups); idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    d = []
    for _ in range(N_BOOT):
        idx = np.concatenate([idx_by[g] for g in rng.choice(uniq, len(uniq))])
        if y[idx].min() != y[idx].max():
            d.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def centralized_xgb(labeled, cols, y, train_idx, val_idx, test_idx):
    import xgboost as xgb
    import psyche_d_treeavg_audit_extend_s4_centralized_xgboost as S4C
    yt, yv = y.loc[train_idx].to_numpy(), y.loc[val_idx].to_numpy()
    params = dict(S4C.XGB_BASE_PARAMS, scale_pos_weight=int((yt == 0).sum()) / max(int((yt == 1).sum()), 1))
    d = lambda idx, lab=None: xgb.DMatrix(labeled.loc[idx, cols].to_numpy(np.float64), label=lab, missing=np.nan)
    dtrain, dval, dtest = d(train_idx, yt), d(val_idx, yv), d(test_idx)
    bst = xgb.train(params, dtrain, num_boost_round=S4C.MAX_BOOST_ROUND, evals=[(dtrain, "train"), (dval, "val")],
                    early_stopping_rounds=S4C.EARLY_STOPPING_ROUNDS, verbose_eval=False)
    rng = (0, bst.best_iteration + 1)
    return {"y_val": yv, "val": bst.predict(dval, iteration_range=rng), "y_test": y.loc[test_idx].to_numpy(),
            "test": bst.predict(dtest, iteration_range=rng), "best_round": bst.best_iteration + 1}


def load_frame():
    import pickle
    cache = pickle.loads(C_SPLIT_CACHE.read_bytes())
    labeled, base_cols, y = C.load_arm_b_frame()
    assert base_cols == cache["feature_cols"]
    idx = {k: labeled.index[labeled["user_id"].isin(cache[f"{k}_participants"])] for k in ("train", "val", "test")}
    e = np.load(EMB_PATH)
    pos = {k: i for i, k in enumerate(e["row_index"])}
    emb = pd.DataFrame(e["emb"][[pos[str(k)] for k in labeled.index]], index=labeled.index,
                       columns=[f"moment_{i}" for i in range(e["emb"].shape[1])])
    labeled = pd.concat([labeled, emb], axis=1)
    cols = list(base_cols) + list(emb.columns)
    assert len(cols) == 664
    return cache, labeled, list(base_cols), cols, y, idx["train"], idx["val"], idx["test"]


def save_part(name, runs):
    flat = {f"{m}|{sd}|{k}": np.asarray(v) for m, per in runs.items() for sd, d in per.items() for k, v in d.items()}
    np.savez_compressed(str(PART_PATH).format(name), **flat)


def load_part(name):
    z = np.load(str(PART_PATH).format(name))
    runs = {}
    for key in z.files:
        m, sd, k = key.split("|")
        runs.setdefault(m, {}).setdefault(sd, {})[k] = z[key]
    return runs


def part_mlp():
    import psyche_d_fedavg_vs_fedtree_mlp_model as M
    import psyche_d_fedavg_vs_fedtree_step23_fedavg as S23
    cache, labeled, base_cols, cols, y, tr, va, te = load_frame()
    M.SmallMLP.__init__.__defaults__ = (len(cols),)  # widen the MLP input for this process only
    runs = {"FedAvg MLP": {}}
    for seed in C.PARTITION_SEEDS:
        r = S23.run_one_seed(seed, cache, labeled, cols, y, tr, va, te)
        runs["FedAvg MLP"][str(seed)] = {"y_val": r["y_val"], "val": r["fedavg_val_predictions"], "y_test": r["y_test"],
                                         "test": r["fedavg_test_predictions"], "best_round": r["fedavg_best_round"]}
        print(f"FedAvg seed {seed} done", flush=True)
    M.SmallMLP.__init__.__defaults__ = (len(base_cols),)  # parity: unchanged 152-input setup, seed 1
    r = S23.run_one_seed(1, cache, labeled, base_cols, y, tr, va, te)
    runs["parity_FedAvg_152"] = {"1": {"test": r["fedavg_test_predictions"]}}
    save_part("mlp", runs)


def part_trees():
    import psyche_d_fedavg_vs_fedtree_step4_fed_xgb_bagging as S4F
    import psyche_d_fedavg_vs_fedtree_step5_tree_avg as S5
    cache, labeled, base_cols, cols, y, tr, va, te = load_frame()
    runs = {"FedXgbBagging": {}, "TreeAvg": {}, "Centralized XGBoost": {}, "parity_TreeAvg_152": {}, "parity_FedXgbBagging_152": {}}
    for seed in C.PARTITION_SEEDS:
        sd = str(seed)
        r = S4F.run_one_seed(seed, cache, labeled, cols, y, tr, va, te)
        runs["FedXgbBagging"][sd] = {"y_val": r["y_val"], "val": r["fed_xgb_bagging_val_predictions"], "y_test": r["y_test"],
                                     "test": r["fed_xgb_bagging_test_predictions"], "best_round": r["best_round"]}
        r = S5.run_one_seed(seed, cache, labeled, cols, y, tr, va, te)
        runs["TreeAvg"][sd] = {"y_val": r["y_val"], "val": r["tree_avg_val_predictions"], "y_test": r["y_test"],
                               "test": r["tree_avg_test_predictions"], "best_round": r["best_round"]}
        print(f"trees seed {seed} done", flush=True)
    # Parity: the unchanged 152-feature setup must reproduce the saved primary predictions (seed 1).
    r = S5.run_one_seed(1, cache, labeled, base_cols, y, tr, va, te)
    runs["parity_TreeAvg_152"]["1"] = {"test": r["tree_avg_test_predictions"]}
    r = S4F.run_one_seed(1, cache, labeled, base_cols, y, tr, va, te)
    runs["parity_FedXgbBagging_152"]["1"] = {"test": r["fed_xgb_bagging_test_predictions"]}
    c = centralized_xgb(labeled, cols, y, tr, va, te)
    runs["Centralized XGBoost"]["1"] = c
    save_part("trees", runs)


def part_report():
    t0 = time.time()
    cache, labeled, base_cols, cols, y, tr, va, te = load_frame()
    groups_test = labeled.loc[te, "user_id"].to_numpy()
    runs = {**load_part("mlp"), **load_part("trees")}
    saved = {
        "FedAvg MLP": {str(r["partition_seed"]): (r["fedavg_val_predictions"], r["fedavg_test_predictions"])
                       for r in S13.load("psyche_d_fedavg_vs_fedtree_step23_results.json")["per_seed_results"]},
        "FedXgbBagging": {str(r["partition_seed"]): (r["fed_xgb_bagging_val_predictions"], r["fed_xgb_bagging_test_predictions"])
                          for r in S13.load("psyche_d_fedavg_vs_fedtree_step4_results.json")["per_seed_results"]},
        "TreeAvg": {str(r["partition_seed"]): (r["tree_avg_val_predictions"], r["tree_avg_test_predictions"])
                    for r in S13.load("psyche_d_fedavg_vs_fedtree_step5_results.json")["per_seed_results"]},
    }
    s4 = S13.load("psyche_d_treeavg_audit_extend_s4_results.json")
    saved["Centralized XGBoost"] = {"1": (s4["centralized_xgboost_val_predictions"], s4["centralized_xgboost_test_predictions"])}

    st5 = S13.load("psyche_d_fedavg_vs_fedtree_step5_results.json")["per_seed_results"][0]["tree_avg_test_predictions"]
    st4 = S13.load("psyche_d_fedavg_vs_fedtree_step4_results.json")["per_seed_results"][0]["fed_xgb_bagging_test_predictions"]
    parity = {"TreeAvg_152_seed1_max_abs_diff": float(np.abs(runs.pop("parity_TreeAvg_152")["1"]["test"] - np.asarray(st5)).max()),
              "FedXgbBagging_152_seed1_max_abs_diff": float(np.abs(runs.pop("parity_FedXgbBagging_152")["1"]["test"] - np.asarray(st4)).max())}
    st23 = S13.load("psyche_d_fedavg_vs_fedtree_step23_results.json")["per_seed_results"][0]["fedavg_test_predictions"]
    parity["FedAvg_152_seed1_max_abs_diff"] = float(np.abs(runs.pop("parity_FedAvg_152")["1"]["test"] - np.asarray(st23)).max())
    print("parity", parity)
    out = {"meta": {"script": "scripts/psyche_d_moment_small_fl.py", "n_features": len(cols),
                    "feature_set": "152 primary features + 512-d frozen MOMENT-Small embedding of the monthly sleep/steps history",
                    "status": "supplementary experiment, not part of the frozen primary results",
                    "treeavg_config": "n_k/N, eta 0.10, 30 rounds, 1 tree/client/round (frozen anchor)",
                    "threshold_rule": "argmax validation balanced accuracy (S13.select_threshold), frozen, applied once to test",
                    "bootstrap": f"participant-level paired, {N_BOOT} resamples, partition seed 1",
                    "runtime_sec": None}, "parity_check": parity, "methods": {}}
    for name, per in runs.items():
        per = {sd: {k: (v.item() if np.ndim(v) == 0 else v) for k, v in d.items()} for sd, d in per.items()}
        rows = {}
        for sd, d in per.items():
            yt = d["y_test"]
            sv, st = (np.asarray(x) for x in saved[name][sd])
            t_with = S13.select_threshold(d["y_val"], d["val"]); t_without = S13.select_threshold(d["y_val"], sv)
            rows[sd] = {"auroc_with": float(roc_auc_score(yt, d["test"])), "auroc_without": float(roc_auc_score(yt, st)),
                        "auprc_with": float(average_precision_score(yt, d["test"])), "auprc_without": float(average_precision_score(yt, st)),
                        "best_round_with": int(d["best_round"]), "op_with": op(yt, d["test"], t_with), "op_without": op(yt, st, t_without)}
            if sd == "1":
                rows[sd]["delta_ci95_paired"] = paired_ci(yt, groups_test, d["test"], st)
        agg = lambda k: {"mean": float(np.mean([rows[s][k] for s in rows])), "sd_pop": float(np.std([rows[s][k] for s in rows]))}
        out["methods"][name] = {"per_seed": rows, "n_runs": len(rows), "auroc_with": agg("auroc_with"), "auroc_without": agg("auroc_without"),
                                "auprc_with": agg("auprc_with"), "auprc_without": agg("auprc_without"),
                                "delta_auroc_mean": float(np.mean([rows[s]["auroc_with"] - rows[s]["auroc_without"] for s in rows])),
                                "delta_ci95_seed1": rows["1"]["delta_ci95_paired"]}
        m = out["methods"][name]
        print(f'{name:20s} with {m["auroc_with"]["mean"]:.4f}±{m["auroc_with"]["sd_pop"]:.4f}  without {m["auroc_without"]["mean"]:.4f}  '
              f'delta {m["delta_auroc_mean"]:+.4f}  seed1 CI {m["delta_ci95_seed1"]}')
    out["meta"]["runtime_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=1)); UI.write_text(json.dumps(out, indent=1))
    print("wrote", OUT, UI)




if __name__ == "__main__":
    {"mlp": part_mlp, "trees": part_trees, "report": part_report}[sys.argv[1]]()
