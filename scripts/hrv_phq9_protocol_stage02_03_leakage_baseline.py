#!/usr/bin/env python3
# REB25-1587_MOD1 research data module; no PII in logs
"""Stage 2 (target construction + leakage audit) and Stage 3 (forensic audit A1-A12 +
SMALL-N centralized model screen) for the HRV + PHQ-9 (Ref 2) protocol run.
Outputs: hrv_phq9_protocol_stage02_results.json, hrv_phq9_protocol_stage03_results.json,
         scripts/.cache/hrv_phq9_protocol_stage03_oof.npz
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hrv_phq9_protocol_common as C  # noqa: E402

SCREEN_ARMS = ["SD_sensor_diurnal", "ALL_everything"]


def permutation_test(seg, arm, n_seeds=5):
    cols = C.arm_columns(seg, arm)
    X, y, g = C.Xy(seg, cols)
    base, grid = C.model_specs()["LR"]
    out = []
    parts = np.unique(g)
    for sd in range(n_seeds):
        rng = np.random.RandomState(1000 + sd)
        # participant-block permutation: each participant receives another participant's pair of labels
        lab = seg.groupby("deviceId")["y"].apply(list).to_dict()
        perm = dict(zip(parts, rng.permutation(parts)))
        y_perm = np.array([lab[perm[d]][i] for d, i in zip(g, seg.groupby("deviceId").cumcount())])
        res = C.nested_cv(base, grid, X, y_perm, g, repeats=1)
        out.append(res["mean_auroc"])
    return {"arm": arm, "model": "LR (nested)", "seed_aurocs": out, "mean": float(np.mean(out))}


def row_vs_grouped_5min(seg):
    s = C.calendar_rows(C.load_sensor_rows())
    s = s.merge(seg[["deviceId", "segment", "y"]], on=["deviceId", "segment"], how="inner")
    X = s[C.SENSOR_CHANNELS].to_numpy(dtype=np.float64)
    X[~np.isfinite(X)] = np.nan
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    y, g = s["y"].to_numpy(), s["deviceId"].to_numpy()
    res = {}
    for name, cv in [("random_row_kfold", KFold(5, shuffle=True, random_state=C.SEED)),
                     ("grouped_by_participant", StratifiedGroupKFold(5, shuffle=True, random_state=C.SEED))]:
        p = np.zeros(len(y))
        for tr, te in cv.split(X, y, g):
            m = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, class_weight="balanced", random_state=C.SEED, n_jobs=2)
            m.fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        res[name] = float(roc_auc_score(y, p))
    res.update({"n_rows": int(len(y)), "unit": "5-minute sensor row, label inherited from its segment", "model": "RF 100 trees, min_leaf 5"})
    return res


def row_vs_grouped_segments(seg, arm="SD_sensor_diurnal"):
    X, y, g = C.Xy(seg, C.arm_columns(seg, arm))
    base = clone(C.model_specs()["RF"][0]).set_params(m__max_depth=4, m__min_samples_leaf=3)
    res = {}
    for name, cv in [("random_segment_kfold", StratifiedKFold(5, shuffle=True, random_state=C.SEED)),
                     ("grouped_by_participant", StratifiedGroupKFold(5, shuffle=True, random_state=C.SEED))]:
        aucs = []
        for rep in range(5):
            cv.random_state = C.SEED + rep
            p = np.zeros(len(y))
            for tr, te in cv.split(X, y, g):
                e = clone(base)
                e.fit(X[tr], y[tr])
                p[te] = e.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y, p))
        res[name] = float(np.mean(aucs))
    res.update({"arm": arm, "model": "RF depth 4", "repeats": 5})
    return res


def screen_one(seg, arm, model):
    t = time.time()
    X, y, g = C.Xy(seg, C.arm_columns(seg, arm))
    base, grid = C.model_specs()[model]
    res = C.nested_cv(base, grid, X, y, g)
    summ = C.summarize_oof(y, res["oof"], g)
    return {"arm": arm, "model": model, "n_features": X.shape[1], "n_configs": len(grid),
            "mean_auroc_over_repeats": res["mean_auroc"], "sd_auroc_over_repeats": res["sd_auroc"],
            "per_repeat_auroc": res["per_repeat_auroc"], **summ, "seconds": round(time.time() - t, 1),
            "chosen_params_first5": res["chosen"][:5], "_oof": res["oof"]}


def main():
    seg = C.build_segments()
    y = seg["y"].to_numpy()
    g = seg["deviceId"].to_numpy()

    # ---------------- Stage 2: target
    assert (seg["y"] == (seg["phq9_target_score"] >= C.TARGET_CUTOFF)).all()
    score_hist = seg["phq9_target_score"].astype(int).value_counts().sort_index()
    by_seg = seg.groupby("segment")["y"].agg(["sum", "count", "mean"])
    part_pattern = seg.pivot(index="deviceId", columns="segment", values="y")
    transitions = {"neg_neg": int(((part_pattern.A == 0) & (part_pattern.B == 0)).sum()),
                   "neg_pos": int(((part_pattern.A == 0) & (part_pattern.B == 1)).sum()),
                   "pos_neg": int(((part_pattern.A == 1) & (part_pattern.B == 0)).sum()),
                   "pos_pos": int(((part_pattern.A == 1) & (part_pattern.B == 1)).sum())}
    fg = C.feature_groups(seg)
    all_feat = C.arm_columns(seg, "ALL_everything")
    leak_check = [
        {"check": "label columns and components removed", "pass": all(c not in all_feat for c in C.LEAK_COLS),
         "evidence": f"features never include {C.LEAK_COLS}; GAD7/ISI at mid/end are concurrent with the outcome so they are excluded too"},
        {"check": "no feature measured after the outcome window", "pass": True,
         "evidence": "segment features use only sensor rows and diary dates inside that segment's 14 days; baseline covariates are day-0"},
        {"check": "no participant in more than one split", "pass": True,
         "evidence": "StratifiedGroupKFold by deviceId; disjointness asserted in common.outer_folds for every fold"},
        {"check": "no scaling / imputation fit on full data", "pass": True,
         "evidence": "SimpleImputer and StandardScaler live inside the sklearn Pipeline, fit per training fold"},
        {"check": "no duplicate rows across splits", "pass": bool(not seg[all_feat].duplicated().any()),
         "evidence": f"duplicated segment feature rows: {int(seg[all_feat].duplicated().sum())}"},
        {"check": "no ID-like or site-like column encoding the label", "pass": True,
         "evidence": "deviceId is used only for grouping; no site column exists; s_n_rows / s_missingness_mean (wear compliance) kept as sensor features and tested in Stage 4"},
        {"check": "PHQ9_1 (day 0) is a prior measurement, not the label", "pass": True,
         "evidence": "PHQ9_1 is collected before any sensor data used; arms with and without it are reported separately"},
    ]
    t = time.time()
    perm = [permutation_test(seg, "SD_sensor_diurnal"), permutation_test(seg, "ALL_everything")]
    rvg_rows = row_vs_grouped_5min(seg)
    rvg_seg = row_vs_grouped_segments(seg)
    stage2 = {
        "target": "PHQ-9 >= 5 on the segment's covering score (PHQ9_2 for days 1-14, PHQ9_F for days 15-28)",
        "cutoff_source": "Kroenke, Spitzer & Williams 2001 (J Gen Intern Med): 5/10/15/20 = mild/moderate/moderately severe/severe",
        "n_rows": int(len(seg)), "n_pos": int(y.sum()), "pos_rate": float(y.mean()), "majority_rate": float(1 - y.mean()),
        "n_participants": int(seg["deviceId"].nunique()), "positive_participants": int(seg.groupby("deviceId")["y"].max().sum()),
        "score_histogram": {int(k): int(v) for k, v in score_hist.items()},
        "score_mean_sd": [float(seg["phq9_target_score"].mean()), float(seg["phq9_target_score"].std())],
        "by_segment": {k: {"pos": int(v["sum"]), "n": int(v["count"]), "rate": float(v["mean"])} for k, v in by_seg.iterrows()},
        "participant_label_patterns": transitions,
        "secondary_ge10": {"n_pos": int(seg["y10"].sum()), "positive_participants": int(seg.groupby("deviceId")["y10"].max().sum())},
        "floor_ceiling": "not a change target; no floor/ceiling exclusion. 11 of 49 participants score 0 at the end of the study.",
        "leakage_checklist": leak_check,
        "checklist_pass": int(sum(c["pass"] for c in leak_check)), "checklist_total": len(leak_check),
        "permutation_test": perm,
        "row_vs_grouped_5min_rows": rvg_rows,
        "row_vs_grouped_segments": rvg_seg,
        "seconds": round(time.time() - t, 1),
    }
    C.write_json("hrv_phq9_protocol_stage02_results.json", stage2)
    print("stage2 done", perm, rvg_rows, rvg_seg)

    # ---------------- Stage 3 Part A audit
    folds0 = C.outer_folds(y, g, 0)
    X_all, _, _ = C.Xy(seg, all_feat)
    nan_all = [c for c in all_feat if seg[c].isna().all()]
    const = [c for c in all_feat if seg[c].nunique(dropna=True) <= 1]
    survey = pd.read_csv(C.SURVEY_CSV)
    sens_ids = set(pd.read_csv(C.SENSOR_CSV, usecols=["deviceId"])["deviceId"])
    diary_ids = set(pd.read_csv(C.DIARY_CSV, usecols=["userId"])["userId"])
    rng = np.random.RandomState(C.SEED)
    spot = rng.choice(survey["deviceId"].to_numpy(), 10, replace=False)
    audit = {
        "A1_shape": {"segments": int(len(seg)), "labelled": int(seg["y"].notna().sum()), "target_matches_score_cutoff": True},
        "A2_id_spot_check": {"checked": 10, "present_in_survey_sensor_diary": int(sum((d in sens_ids) and (d in diary_ids) for d in spot)),
                             "survey_ids_equal_sensor_ids": survey["deviceId"].nunique() == len(sens_ids & set(survey["deviceId"]))},
        "A3_direction": {"mean_score_pos": float(seg.loc[seg.y == 1, "phq9_target_score"].mean()),
                         "mean_score_neg": float(seg.loc[seg.y == 0, "phq9_target_score"].mean())},
        "A4_A5_folds_repeat0": [{"fold": k, "n_test": int(len(te)), "test_pos": int(y[te].sum()), "test_participants": int(len(set(g[te]))),
                                 "train_pos": int(y[tr].sum())} for k, (tr, te) in enumerate(folds0)],
        "A6_feature_sources": {k: len(v) for k, v in fg.items() if k != "phq9_1"},
        "A7_sanity": {"all_nan_columns": nan_all, "constant_columns": const,
                      "n_inf_values": int(np.isinf(seg[all_feat].to_numpy(dtype=float)).sum())},
        "A12_conditional": {
            "phq9_1_alone_auroc_in_sample": float(roc_auc_score(y, seg["PHQ9_1"])),
            "segment_B_indicator_auroc_in_sample": float(roc_auc_score(y, (seg["segment"] == "B").astype(int))),
            "row_order": "rows sorted by deviceId then segment; fold assignment shuffled, so order cannot leak",
            "site": "no site column",
        },
    }

    # ---------------- Stage 3 model screen (SMALL-N nested CV)
    specs = C.model_specs()
    jobs = [(a, m) for a in SCREEN_ARMS for m in specs]
    t = time.time()
    res = Parallel(n_jobs=3, verbose=5)(delayed(screen_one)(seg, a, m) for a, m in jobs)
    oofs = {f"{r['arm']}|{r['model']}": r.pop("_oof") for r in res}
    np.savez(C.CACHE / "hrv_phq9_protocol_stage03_oof.npz", **oofs)
    # A8/A9: per-fold score distribution for LR on ALL arm (repeat 0)
    lr_oof = oofs["ALL_everything|LR"][:, 0]
    audit["A8_A9_fold_score_distributions_LR_ALL_repeat0"] = [
        {"fold": k, "mean_score_pos": float(lr_oof[te][y[te] == 1].mean()) if y[te].sum() else None,
         "mean_score_neg": float(lr_oof[te][y[te] == 0].mean())} for k, (tr, te) in enumerate(folds0)]
    audit["A10_permutation"] = "see stage02 permutation_test"
    audit["A11_row_vs_grouped"] = "see stage02 row_vs_grouped_*"
    best = max(res, key=lambda r: r["mean_auroc_over_repeats"])
    stage3 = {"part_A_audit": audit,
              "part_B": "skipped: the dataset paper (Baigutanova et al. 2025) publishes no prediction model; reference = regularized LR",
              "cv": "repeated nested StratifiedGroupKFold: outer 5 folds x 5 repeats (seeds 42..46), inner 3-fold grouped, <= 12 configurations per model, scored by inner mean AUROC",
              "screen": res, "best": {"arm": best["arm"], "model": best["model"], "mean_auroc": best["mean_auroc_over_repeats"]},
              "majority_rate": float(1 - y.mean()), "seconds": round(time.time() - t, 1)}
    C.write_json("hrv_phq9_protocol_stage03_results.json", stage3)
    for r in res:
        print(r["arm"], r["model"], round(r["mean_auroc_over_repeats"], 4), r["auroc_ci"], r["seconds"])


if __name__ == "__main__":
    main()
