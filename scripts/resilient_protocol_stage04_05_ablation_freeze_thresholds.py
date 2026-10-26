#!/usr/bin/env python3
# RESILIENT dataset (Zenodo 16755408): IRAS 321104, independent of REB25-1587_MOD1. Research data.
"""Protocol Stage 4 (feature-source ablation, freeze, importance) and Stage 5
(validation-selected operating points). Run with .venv-psyche-d-fl-mlp.

Freeze rule (written before Stage 4 results were seen): arm = sensor + demographics
(deployment-realistic for a wearable Agent 1); model family = best mean per-repeat AUROC
in the Stage 3 screen; hyperparameters are nested-selected per outer fold during
evaluation, and the frozen deployable config is the one inner 3-fold CV selects on all
73 participants. Temporal validity (Stage 4.3) is skipped: one baseline PHQ-9 per
participant, no later labels.

Stage 5: thresholds are chosen ONLY on inner out-of-fold scores of each outer training
fold, then applied once to that outer test fold; predictions are pooled over all 25
outer folds (each participant appears once per repeat). Reference sensitivity for the
matched point = clinical target 0.80 (no published model on this data, P8 = no).

Writes scripts/resilient_protocol_stage04_05_results.json and
scripts/resilient_protocol_frozen_config.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resilient_protocol_common as C  # noqa: E402

OUT = C.SCRIPT_DIR / "resilient_protocol_stage04_05_results.json"
FROZEN = C.SCRIPT_DIR / "resilient_protocol_frozen_config.json"
S23 = C.SCRIPT_DIR / "resilient_protocol_stage02_03_results.json"


def main():
    d = C.load_features()
    F = d["features"]
    uids = F.index.to_numpy()
    y = d["y_phq10"]
    g = d["sensor_groups"]
    sensor = [c for cs in g.values() for c in cs]
    demo, quest = d["demo_cols"], d["quest_cols"]
    s23 = json.loads(S23.read_text())
    model = s23["screen_best_by_mean_repeat_auroc"]
    print("[freeze] model family from Stage 3 screen:", model)

    arms = {
        "sensor_demo (candidate)": sensor + demo,
        "sensor_only": sensor,
        "watch_only (HR+steps)": g["watch_hr"] + g["watch_steps"],
        "mat_only (sleep physio+state)": g["mat_physio"] + g["mat_state"],
        "demographics_only": demo,
        "questionnaire_only (GAD-7, ACE-III)": quest,
        "everything (sensor+demo+questionnaire)": sensor + demo + quest,
    }
    res = {"model": model, "arms": {}}
    oofs = {}
    frozen_run = None
    for arm, cols in arms.items():
        X = F[cols].to_numpy(dtype=float)
        r = C.repeated_nested_cv(model, X, y, uids, seed=C.SEED, with_thresholds=arm.startswith("sensor_demo"))
        s = C.summarize(y, r["oof"])
        s["n_features"] = len(cols)
        s["columns"] = cols
        res["arms"][arm] = s
        oofs[arm] = r["oof"].mean(axis=0)
        if arm.startswith("sensor_demo"):
            frozen_run = r
            screen_oof = np.array(s23["screen"][model]["oof"])
            res["reproduces_stage3_oof"] = bool(np.allclose(screen_oof, r["oof"], atol=1e-6))
        print(f"[arm] {arm}: {s['auroc']:.3f} [{s['auroc_ci'][0]:.3f},{s['auroc_ci'][1]:.3f}]", flush=True)
    cand = oofs["sensor_demo (candidate)"]
    res["paired_vs_candidate"] = {a: C.paired_boot_diff(y, oofs[a], cand) for a in arms if not a.startswith("sensor_demo")}
    res["temporal_validity"] = "skipped: one baseline PHQ-9 per participant (P3), no later labels"

    # label-shuffle permutation test with the FROZEN model family (5 seeds, 1 repeat each)
    Xc = F[arms["sensor_demo (candidate)"]].to_numpy(dtype=float)
    perm = []
    for s in range(5):
        yp = np.random.default_rng(2000 + s).permutation(y)
        rp = C.repeated_nested_cv(model, Xc, yp, uids, seed=C.SEED, n_repeats=1)
        perm.append(float(roc_auc_score(yp, rp["oof"][0])))
    res["permutation_frozen_model"] = {"aurocs": perm, "mean": float(np.mean(perm)), "sd": float(np.std(perm))}
    print("[perm frozen]", res["permutation_frozen_model"], flush=True)

    # ---------------- freeze ----------------
    cols = arms["sensor_demo (candidate)"]
    X = F[cols].to_numpy(dtype=float)
    params, inner_auc, _ = C.nested_select(model, X, y, C.SEED)
    _, est = C._fit_predict(model, params, X, y, X, C.SEED)
    frozen = {"arm": "sensor + demographics", "model": model, "n_features": len(cols), "features": cols,
              "hyperparameters_full_data_inner_cv": params, "inner_cv_auroc_full_data": inner_auc,
              "evaluation": "repeated nested stratified CV, outer 5-fold x 5 repeats (seeds 42-46), inner 3-fold, <=10 configs",
              "seed": C.SEED, "class_imbalance": "balanced class weights / scale_pos_weight / is_unbalance, no SMOTE",
              "auroc": res["arms"]["sensor_demo (candidate)"]["auroc"], "auroc_ci": res["arms"]["sensor_demo (candidate)"]["auroc_ci"]}
    FROZEN.write_text(json.dumps(frozen, indent=2, default=C._json_default))
    res["frozen"] = frozen

    # importance: model-native (full-data fit) + permutation on outer test folds of repeat 0
    native = None
    clf = est.named_steps["clf"] if hasattr(est, "named_steps") else est
    if hasattr(clf, "coef_"):
        native = {"type": "standardized LR coefficient", "values": dict(zip(cols, clf.coef_.ravel().tolist()))}
    elif hasattr(clf, "feature_importances_"):
        native = {"type": "impurity/gain importance", "values": dict(zip(cols, np.asarray(clf.feature_importances_, float).tolist()))}
    res["importance_native"] = native
    rng = np.random.default_rng(C.SEED)
    from sklearn.model_selection import StratifiedKFold
    outer = StratifiedKFold(C.N_OUTER, shuffle=True, random_state=C.SEED)
    fitted = []
    base = np.zeros(len(y))
    for f, (tr, te) in enumerate(outer.split(X, y)):
        p, _ = C.nested_select(model, X[tr], y[tr], C.SEED + f)[0], None
        prob, m = C._fit_predict(model, p, X[tr], y[tr], X[te], C.SEED + f)
        base[te] = prob
        fitted.append((m, te))
    base_auc = roc_auc_score(y, base)
    perm_imp = {}
    for j, c in enumerate(cols):
        drops = []
        for _ in range(10):
            s = np.zeros(len(y))
            for m, te in fitted:
                Xp = X[te].copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                s[te] = m.predict_proba(Xp)[:, 1]
            drops.append(base_auc - roc_auc_score(y, s))
        perm_imp[c] = float(np.mean(drops))
    res["importance_permutation"] = {"base_auroc_repeat0": base_auc, "auroc_drop_mean_of_10": perm_imp}

    # ---------------- Stage 5 operating points ----------------
    ys, ss, th = [], [], {"default": [], "max_accuracy": [], "matched_sensitivity": []}
    pooled = {k: {"y": [], "pred": []} for k in th}
    for fr in frozen_run["folds"]:
        te = np.array(fr["test_idx"])
        sc = np.array(fr["test_scores"])
        for k in th:
            t = fr["thresholds"][k]
            th[k].append(t)
            pooled[k]["y"] += y[te].tolist()
            pooled[k]["pred"] += (sc >= t).astype(int).tolist()
        ys += y[te].tolist()
        ss += sc.tolist()
    ops = {}
    for k in th:
        yy, pp = np.array(pooled[k]["y"]), np.array(pooled[k]["pred"])
        tp = int(((pp == 1) & (yy == 1)).sum()); fn = int(((pp == 0) & (yy == 1)).sum())
        tn = int(((pp == 0) & (yy == 0)).sum()); fp = int(((pp == 1) & (yy == 0)).sum())
        ops[k] = {"threshold_median": float(np.median(th[k])), "threshold_iqr": [float(np.percentile(th[k], 25)), float(np.percentile(th[k], 75))],
                  "accuracy": (tp + tn) / len(yy), "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
                  "ppv": tp / (tp + fp) if tp + fp else None, "tp": tp, "fn": fn, "tn": tn, "fp": fp, "n_predictions": int(len(yy))}
    ops["matched_sensitivity"]["reference_sensitivity"] = C.CLINICAL_TARGET_SENS
    ops["matched_sensitivity"]["reference_source"] = "clinical screening target 0.80 (protocol fallback; no published model on RESILIENT)"
    res["operating_points"] = ops
    res["majority_rate"] = float(1 - y.mean())
    s_avg = frozen_run["oof"].mean(axis=0)
    res["sweep_repeat_averaged_oof"] = [C.confusion_at(y, s_avg, t) for t in np.round(np.arange(0.01, 1.0, 0.01), 2)]
    res["target_accuracy"] = "no target accuracy was set for this dataset; the majority rate is the bar"
    res["frozen_oof"] = frozen_run["oof"].tolist()
    res["arm_oof_avg"] = {a: v.tolist() for a, v in oofs.items()}
    print(json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ("threshold_median", "accuracy", "sensitivity", "specificity")} for k, v in ops.items()}))
    C.write_json(OUT, res)


if __name__ == "__main__":
    main()
