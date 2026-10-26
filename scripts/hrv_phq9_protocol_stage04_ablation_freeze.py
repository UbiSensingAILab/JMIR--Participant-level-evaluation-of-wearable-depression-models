#!/usr/bin/env python3
# REB25-1587_MOD1 research data module; no PII in logs
"""Stage 4: feature-source ablation, temporal validity check, baseline freeze, feature importance.
SMALL-N branch: every arm under the same repeated nested grouped CV (5x5, inner 3-fold).
Freeze rule (fixed before looking at the arm results): the arm x model with the highest
mean repeated-CV AUROC; its hyperparameters = the modal inner-CV choice over the 25 outer folds.
Outputs: hrv_phq9_protocol_stage04_results.json, hrv_phq9_protocol_frozen_config.json,
         scripts/.cache/hrv_phq9_protocol_stage04_oof.npz
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hrv_phq9_protocol_common as C  # noqa: E402


def run_arm(seg, arm, model):
    X, y, g = C.Xy(seg, C.arm_columns(seg, arm))
    base, grid = C.model_specs()[model]
    res = C.nested_cv(base, grid, X, y, g)
    return {"arm": arm, "model": model, "n_features": X.shape[1], "mean_auroc": res["mean_auroc"], "sd_auroc": res["sd_auroc"],
            "per_repeat_auroc": res["per_repeat_auroc"], **C.summarize_oof(y, res["oof"], g),
            "chosen": res["chosen"], "_oof": res["oof"]}


def temporal(seg, arm, model, params):
    """(i) train on segment A, test on segment B of the same people (time only);
    (ii) train on A of training participants, test on B of held-out participants (time + people)."""
    cols = C.arm_columns(seg, arm)
    base = clone(C.model_specs()[model][0]).set_params(**params)
    A, B = seg[seg.segment == "A"].reset_index(drop=True), seg[seg.segment == "B"].reset_index(drop=True)
    XA, yA, gA = C.Xy(A, cols)
    XB, yB, gB = C.Xy(B, cols)
    e = clone(base)
    C._fit(e, XA, yA)
    same_people = C.safe_auc(yB, C._proba(e, XB))
    aucs = []
    for rep in range(C.OUTER_REPEATS):
        p = np.full(len(yB), np.nan)
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=C.SEED + rep)
        for tr, te in cv.split(XB, yB, gB):
            trp = set(gB[tr])
            assert trp.isdisjoint(set(gB[te]))
            mA = np.isin(gA, list(trp))
            e = clone(base)
            C._fit(e, XA[mA], yA[mA])
            p[te] = C._proba(e, XB[te])
        aucs.append(C.safe_auc(yB, p))
    return {"train_A_test_B_same_people_auroc": same_people,
            "train_A_test_B_disjoint_people_auroc_mean": float(np.mean(aucs)), "per_repeat": aucs,
            "n_A_pos": int(yA.sum()), "n_B_pos": int(yB.sum())}


def main():
    t = time.time()
    seg = C.build_segments()
    s3 = C.load_json("hrv_phq9_protocol_stage03_results.json")
    # best tree-type model on the wearable arm from the Stage 3 screen (LR is always run too)
    tree_models = ["RF", "ET", "HGB", "LGBM_DART", "XGB"]
    sd_rows = [r for r in s3["screen"] if r["arm"] == "SD_sensor_diurnal" and r["model"] in tree_models]
    best_tree = max(sd_rows, key=lambda r: r["mean_auroc_over_repeats"])["model"]
    models = ["LR", best_tree]
    jobs = [(a, m) for a in C.ARMS for m in models]
    res = Parallel(n_jobs=3, verbose=5)(delayed(run_arm)(seg, a, m) for a, m in jobs)
    oofs = {f"{r['arm']}|{r['model']}": r.pop("_oof") for r in res}
    np.savez(C.CACHE / "hrv_phq9_protocol_stage04_oof.npz", **oofs)

    best = max(res, key=lambda r: r["mean_auroc"])
    wear = [r for r in res if r["arm"] in ("S_sensor", "SD_sensor_diurnal")]
    best_wear = max(wear, key=lambda r: r["mean_auroc"])
    y = seg["y"].to_numpy()
    g = seg["deviceId"].to_numpy()
    cand_key = f"{best['arm']}|{best['model']}"
    pc = np.nanmean(oofs[cand_key], axis=1)
    for r in res:
        k = f"{r['arm']}|{r['model']}"
        r["paired_vs_candidate"] = None if k == cand_key else C.paired_boot_diff(y, np.nanmean(oofs[k], axis=1), pc, g)
    # paired: wearable-best vs PHQ9_1-only LR (does the watch add anything beyond the prior PHQ-9?)
    wk = f"{best_wear['arm']}|{best_wear['model']}"
    sdq = "SDQ_sensor_diurnal_baseline|LR"
    key_contrasts = {
        "best_wearable_vs_phq9_1_only_LR": C.paired_boot_diff(y, np.nanmean(oofs[wk], axis=1), np.nanmean(oofs["P_phq9_1_only|LR"], axis=1), g),
        "sensor_diurnal_baseline_LR_vs_baseline_only_LR": C.paired_boot_diff(y, np.nanmean(oofs[sdq], axis=1), np.nanmean(oofs["Q_baseline_questionnaires_demo|LR"], axis=1), g),
        "baseline_only_LR_vs_phq9_1_only_LR": C.paired_boot_diff(y, np.nanmean(oofs["Q_baseline_questionnaires_demo|LR"], axis=1), np.nanmean(oofs["P_phq9_1_only|LR"], axis=1), g),
    }

    # freeze: modal hyperparameters of the winning arm x model
    modal = Counter(json.dumps(c["params"], sort_keys=True) for c in best["chosen"]).most_common(1)[0]
    frozen_params = json.loads(modal[0])
    cols = C.arm_columns(seg, best["arm"])
    X, _, _ = C.Xy(seg, cols)
    base = C.model_specs()[best["model"]][0]
    fixed = C.nested_cv(base, [frozen_params], X, y, g, fixed_params=frozen_params)
    fixed_sum = C.summarize_oof(y, fixed["oof"], g)
    np.savez(C.CACHE / "hrv_phq9_protocol_frozen_oof.npz", oof=fixed["oof"])

    # importance: coefficients/gain on the full data + grouped permutation importance on outer test folds
    e = clone(base).set_params(**frozen_params)
    C._fit(e, X, y)
    last = e.steps[-1][1]
    if hasattr(last, "coef_"):
        kept = e.named_steps["imp"].get_feature_names_out(cols) if hasattr(e.named_steps["imp"], "get_feature_names_out") else cols
        names = [cols[int(str(n).replace("x", ""))] if str(n).startswith("x") else str(n) for n in kept]
        imp_model = sorted(zip(names, last.coef_[0].tolist()), key=lambda z: -abs(z[1]))[:15]
        imp_kind = "standardized LR coefficient (full-data fit, descriptive only)"
    else:
        imp_model = sorted(zip(cols, last.feature_importances_.tolist()), key=lambda z: -z[1])[:15]
        imp_kind = "impurity / gain importance (full-data fit, descriptive only)"
    perm = np.zeros(len(cols))
    nf = 0
    for tr, te in C.outer_folds(y, g, 0):
        if len(np.unique(y[te])) < 2:
            continue
        e2 = clone(base).set_params(**frozen_params)
        C._fit(e2, X[tr], y[tr])
        pi = permutation_importance(e2, X[te], y[te], scoring="roc_auc", n_repeats=20, random_state=C.SEED)
        perm += pi.importances_mean
        nf += 1
    perm /= max(nf, 1)
    perm_top = sorted(zip(cols, perm.tolist()), key=lambda z: -z[1])[:15]

    temporal_res = temporal(seg, best["arm"], best["model"], frozen_params)
    temporal_wear = temporal(seg, best_wear["arm"], best_wear["model"],
                             json.loads(Counter(json.dumps(c["params"], sort_keys=True) for c in best_wear["chosen"]).most_common(1)[0][0]))

    frozen = {"arm": best["arm"], "model": best["model"], "features": cols, "n_features": len(cols),
              "hyperparameters": frozen_params, "hyperparameter_modal_count": f"{modal[1]}/25 outer folds",
              "seed": C.SEED, "cv": "repeated grouped StratifiedGroupKFold 5 folds x 5 repeats (seeds 42..46)",
              "target": "PHQ-9 >= 5", "nested_cv_mean_auroc": best["mean_auroc"],
              "fixed_params_cv": {"mean_auroc": fixed["mean_auroc"], "sd_auroc": fixed["sd_auroc"],
                                  "per_repeat_auroc": fixed["per_repeat_auroc"], **fixed_sum},
              "freeze_rule": "highest mean repeated nested-CV AUROC across all arm x model cells; modal inner-CV hyperparameters"}
    C.write_json("hrv_phq9_protocol_frozen_config.json", frozen)
    for r in res:
        r.pop("chosen")
    out = {"models": models, "best_tree_from_stage3": best_tree, "arms": {k: v for k, v in C.ARMS.items()},
           "results": res, "candidate": cand_key, "best_wearable_only": wk, "key_contrasts": key_contrasts,
           "frozen": frozen, "importance_model": {"kind": imp_kind, "top": imp_model},
           "importance_permutation_grouped": {"kind": "permutation AUROC drop on outer test folds, repeat 0, 20 shuffles", "top": perm_top},
           "temporal_frozen": temporal_res, "temporal_best_wearable": temporal_wear,
           "majority_rate": float(1 - y.mean()), "seconds": round(time.time() - t, 1)}
    C.write_json("hrv_phq9_protocol_stage04_results.json", out)
    for r in res:
        print(r["arm"], r["model"], round(r["mean_auroc"], 4), r["auroc_ci"])
    print("FROZEN", frozen["arm"], frozen["model"], frozen_params, fixed["mean_auroc"], fixed_sum["auroc_ci"])
    print(key_contrasts)
    print(temporal_res, temporal_wear)


if __name__ == "__main__":
    main()
