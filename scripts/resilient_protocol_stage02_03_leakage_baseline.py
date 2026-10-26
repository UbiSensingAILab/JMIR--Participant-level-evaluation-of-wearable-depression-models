#!/usr/bin/env python3
# RESILIENT dataset (Zenodo 16755408): IRAS 321104, independent of REB25-1587_MOD1. Research data.
"""Protocol Stage 2 (target + leakage audit) and Stage 3 (forensic audit A1-A12 +
centralized model screen, SMALL-N branch). Run with .venv-psyche-d-fl-mlp.

Pre-registered before any model was fit:
- Target: PHQ-9 total >= 10 (Kroenke et al. 2001, moderate or worse; also the locked
  cut-off of train_resilient_multimodal_binary_phq9.py).
- Screen feature set = "sensor + demographics" (24 wearable/mat summaries + 4
  demographic fields). GAD-7 / ACE-III are same-visit questionnaires; they are an
  ablation arm in Stage 4 only, never in the frozen candidate (not available to a
  wearable-only Agent 1 and GAD-7 is strongly collinear with PHQ-9).
- Model screen: 7 families, nested inner 3-fold AUROC, <= 10 sampled configs each,
  outer 5-fold x 5 repeats, seed 42.
- Stage 3 Part B: skipped (P8 = no published model on this data); the reference is the
  strongest simple baseline, regularized logistic regression.

Writes scripts/resilient_protocol_stage02_03_results.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resilient_protocol_common as C  # noqa: E402

OUT = C.SCRIPT_DIR / "resilient_protocol_stage02_03_results.json"
LABEL_LIKE = ("phq", "gds")


def main():
    t_start = time.time()
    d = C.load_features()
    F = d["features"]
    uids = F.index.to_numpy()
    y = d["y_phq10"]
    sensor = [c for g in d["sensor_groups"].values() for c in g]
    cols = sensor + d["demo_cols"]
    X = F[cols].to_numpy(dtype=float)
    res = {}

    # ---------------- Stage 2: target ----------------
    phq = F["phq_total"].to_numpy()
    res["target"] = {
        "definition": "PHQ-9 total >= 10", "cutoff_source": "Kroenke, Spitzer & Williams 2001 (J Gen Intern Med); same cut-off as the existing RESILIENT binary script",
        "raw_score": {"mean": float(phq.mean()), "sd": float(phq.std()), "median": float(np.median(phq)), "min": int(phq.min()), "max": int(phq.max()),
                      "hist": {int(k): int((phq == k).sum()) for k in np.unique(phq)}},
        "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "pos_rate": float(y.mean()), "majority_rate": float(1 - y.mean()),
        "phq_ge5_n": int((phq >= 5).sum()), "gds_ge5_n": int(d["y_gds5"].sum()),
        "floor_ceiling": "not applicable (not a change target)",
    }

    # ---------------- Stage 2: leakage checklist ----------------
    label_cols_in_X = [c for c in cols if c.startswith(LABEL_LIKE)]
    dup_rows = int(F[cols].round(8).duplicated().sum())
    id_like = [c for c in cols if c in ("uid", "user_id", "phq_date_ord")]
    date_auc = float(roc_auc_score(y, F["phq_date_ord"].to_numpy()))
    order_auc = float(roc_auc_score(y, np.arange(len(y))))
    checklist = [
        {"item": "label columns and components removed", "pass": not label_cols_in_X,
         "evidence": f"feature columns starting with phq/gds: {label_cols_in_X or 'none'}; GDS-15 (concurrent depression scale) and GAD-7 excluded from the candidate set"},
        {"item": "no feature measured after the outcome window", "pass": False,
         "evidence": "FAIL by design (disclosed): sensors cover [phq_date, +512 h). Stage 1 found 0/73 participants with any data before phq_date, so a pre-assessment window is impossible. Sensors never see the label, so this is not label leakage, but the model describes a concurrent state, not a prediction ahead of time."},
        {"item": "no participant in more than one split", "pass": True, "evidence": "asserted in code for every outer fold (resilient_protocol_common.repeated_nested_cv)"},
        {"item": "no encoding/scaling/imputation fit on full data", "pass": True, "evidence": "imputer and scaler live inside sklearn Pipelines fit on the training fold only; XGBoost/LightGBM/HistGB use native NaN"},
        {"item": "no duplicate rows across splits", "pass": dup_rows == 0, "evidence": f"{dup_rows} duplicated feature rows"},
        {"item": "no ID-like or site-like column that encodes the label", "pass": not id_like,
         "evidence": f"id/date columns in X: {id_like or 'none'}; single site. Check: AUROC of enrolment date vs label {date_auc:.3f}, of file order vs label {order_auc:.3f} (not used as features)"},
    ]
    res["leakage_checklist"] = checklist
    res["leakage_pass"] = int(sum(c["pass"] for c in checklist))

    # ---------------- Stage 2: permutation test (5 seeds, LogReg, 2 repeats) ----------------
    perm = []
    for s in range(5):
        yp = np.random.default_rng(1000 + s).permutation(y)
        r = C.repeated_nested_cv("LogReg", X, yp, uids, seed=C.SEED, n_repeats=2)
        perm.append(float(roc_auc_score(yp, r["oof"].mean(axis=0))))
    res["permutation_test"] = {"model": "LogReg (nested), outer 5-fold x 2 repeats", "aurocs": perm, "mean": float(np.mean(perm)), "sd": float(np.std(perm))}
    res["row_vs_grouped"] = "not applicable: one row per participant (P2 = 1), so a row split equals a participant split"
    print("[perm]", res["permutation_test"])

    # ---------------- Stage 3 Part A ----------------
    raw_ids_ok = all(isinstance(u, str) and len(u) == 8 for u in uids)
    sens_cols = sensor
    Xdf = F[cols]
    res["audit_A"] = {
        "A1": {"shape": list(Xdf.shape), "labelled_rows": int(len(y)), "target_matches_raw": bool(((phq >= 10).astype(int) == y).all())},
        "A2": {"id_rule": "8-char folder name == Demographics user_id", "spot_check_10_ok": raw_ids_ok and set(uids[:10]) <= set(F.index), "n_unique": int(len(set(uids)))},
        "A3": {"mean_phq_pos": float(phq[y == 1].mean()), "mean_phq_neg": float(phq[y == 0].mean()), "direction_ok": bool(phq[y == 1].min() >= 10 > phq[y == 0].max())},
        "A6": {"sensor_derived": len(sens_cols), "demographic": len(d["demo_cols"]), "questionnaire_excluded": d["quest_cols"]},
        "A7": {"constant_cols": [c for c in cols if Xdf[c].nunique(dropna=True) <= 1], "inf_values": int(np.isinf(X[~np.isnan(X)]).sum()),
               "all_nan_cols": [c for c in cols if Xdf[c].isna().all()]},
        "A11": "not applicable (one row per participant)",
        "A12": {"baseline_severity": "not applicable (no change target)", "enrolment_date_auroc_vs_label": date_auc,
                "file_order_auroc_vs_label": order_auc, "site": "single site"},
    }

    # ---------------- Stage 3 model screen ----------------
    screen = {}
    for name in C.MODEL_NAMES:
        t0 = time.time()
        r = C.repeated_nested_cv(name, X, y, uids, seed=C.SEED)
        s = C.summarize(y, r["oof"])
        s["seconds"] = round(time.time() - t0, 1)
        s["folds"] = [{k: v for k, v in f.items() if k in ("repeat", "fold", "n_train", "n_test", "pos_train", "pos_test", "test_auc", "inner_auc", "params", "score_mean_pos", "score_mean_neg")} for f in r["folds"]]
        s["oof"] = r["oof"].tolist()
        screen[name] = s
        print(f"[screen] {name}: AUROC {s['auroc']:.3f} [{s['auroc_ci'][0]:.3f},{s['auroc_ci'][1]:.3f}] per-rep {s['auroc_per_repeat_mean']:.3f}+-{s['auroc_per_repeat_sd']:.3f} ({s['seconds']}s)", flush=True)
    res["screen"] = screen
    f0 = screen["LogReg"]["folds"]
    res["audit_A"]["A4_A5"] = [{k: f[k] for k in ("repeat", "fold", "n_train", "n_test", "pos_train", "pos_test")} for f in f0]
    res["audit_A"]["A8_A9"] = {name: {"mean_score_pos_by_fold": [r4 for r4 in [f["score_mean_pos"] for f in screen[name]["folds"][:5]]],
                                      "mean_score_neg_by_fold": [f["score_mean_neg"] for f in screen[name]["folds"][:5]]} for name in screen}
    res["audit_A"]["A10"] = res["permutation_test"]
    # Paired comparisons vs the LogReg reference
    ref = np.array(screen["LogReg"]["oof"]).mean(axis=0)
    res["screen_vs_logreg"] = {n: C.paired_boot_diff(y, np.array(screen[n]["oof"]).mean(axis=0), ref) for n in screen if n != "LogReg"}
    best = max(screen, key=lambda n: screen[n]["auroc_per_repeat_mean"])
    res["screen_best_by_mean_repeat_auroc"] = best
    res["feature_columns"] = cols
    res["cv"] = "repeated nested stratified CV: outer 5-fold x 5 repeats (seeds 42..46), inner 3-fold, <=10 configs, participant = row"
    res["seconds_total"] = round(time.time() - t_start, 1)
    C.write_json(OUT, res)


if __name__ == "__main__":
    main()
