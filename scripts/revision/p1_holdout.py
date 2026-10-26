#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. `research_datasets/` is gitignored.
"""JMIR revision Phase 1, held-out analyses on the fixed FL split (tasks 1.4, 1.5, 1.6).

Split: scripts/psyche_d_fedavg_vs_fedtree_split_cache.pkl (seed 20260928; 70/15/15 by
participant; test 1631 rows / 606 participants).
1.4  development = train + validation (85%). Input set x model family chosen by
     pooled out-of-fold AUROC from StratifiedGroupKFold(5, shuffle, 42) on development
     only, refit on all development rows, scored once on test.
1.5  thresholds need an untouched validation set, so the frozen model (152 inputs,
     LightGBM DART) is fitted on the 70% training portion only; thresholds are chosen
     on validation and scored once on test.
1.6  calibration of that same model on test; Platt recalibration fitted on validation.
CIs: participant bootstrap on the test set, 2000 draws, seed 42, shared by all models.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import audit_psyche_d_baseline_reproduction as REPRO  # noqa: E402
import crossfit_psyche_d_two_stage_pipeline as CROSSFIT  # noqa: E402
import rev_common as RC  # noqa: E402
from psyche_d_fedavg_vs_fedtree_step4_fed_xgb_bagging import load_split_and_frame  # noqa: E402

t0 = time.time()
raw = REPRO.load_raw()
phase1_cols = REPRO.build_phase1_input_cols(raw)
base = [c for c in raw.columns if c not in REPRO.LEAKAGE_COLS and c not in ("user_id", "user_mth", "ref_label")]
fmap = pd.read_csv(RC.OUT / "P0_4_psyche_d_feature_groups.csv").set_index("feature")["group"]
WEAR = [c for c in base if fmap[c] == "wearable"]
CTX = [c for c in base if fmap[c] in ("screener", "lifestyle_medication")]
PHQ = ["phq9_score_start", "phq9_cat_start"]
INPUTS = {"a_phq_only": PHQ, "b_wearable_only": WEAR, "c_context_only": CTX, "d_phq_wearable": PHQ + WEAR,
          "e_phq_context": PHQ + CTX, "f_frozen_152": base + PHQ}
FAMILIES = {
    "lightgbm_dart_frozen_config": lambda: REPRO.phase2_dart_factory(),
    "lightgbm_gbdt": lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=42, verbosity=-1, n_jobs=-1),
    "xgboost": lambda: xgb.XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                                         tree_method="hist", random_state=42, n_jobs=-1),
    "logistic_regression": lambda: Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=3000, random_state=42))]),
}

cache, labeled, fcols, yser, train_idx, val_idx, test_idx = load_split_and_frame()
lab = raw[raw["ref_label"]]
assert labeled.index.equals(lab.index) or set(labeled.index) == set(lab.index)
y_all = ((lab["phq9_cat_end"] - lab["phq9_cat_start"]) > 0).astype(int)
dev_idx = train_idx.append(val_idx)


def fit_predict(fam, cols, tr, te_list):
    imp = SimpleImputer(strategy="mean", keep_empty_features=True)
    Xtr = imp.fit_transform(lab.loc[tr, cols].to_numpy(float))
    clf = FAMILIES[fam]()
    sw = compute_sample_weight("balanced", y_all.loc[tr])
    if fam == "logistic_regression":
        clf.fit(Xtr, y_all.loc[tr], lr__sample_weight=sw)
    else:
        clf.fit(Xtr, y_all.loc[tr], sample_weight=sw)
    return [clf.predict_proba(imp.transform(lab.loc[te, cols].to_numpy(float)))[:, 1] for te in te_list]


# ---------------- 1.4 selection on development only ----------------
gd = lab.loc[dev_idx, "user_id"].to_numpy()
yd = y_all.loc[dev_idx].to_numpy()
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.zeros(len(yd)), yd, gd))
sel = []
for iname, cols in INPUTS.items():
    for fam in FAMILIES:
        oof = np.full(len(yd), np.nan)
        for tr, te in folds:
            oof[te] = fit_predict(fam, cols, dev_idx[tr], [dev_idx[te]])[0]
        sel.append({"inputs": iname, "n_inputs": len(cols), "family": fam, "dev_cv_auroc": float(roc_auc_score(yd, oof))})
    print(f"selection {iname} done {time.time() - t0:.0f}s", flush=True)
sel_df = pd.DataFrame(sel).sort_values("dev_cv_auroc", ascending=False)
sel_df.to_csv(RC.OUT / "P1_4_dev_cv_selection.csv", index=False)
best = sel_df.iloc[0]

yt = y_all.loc[test_idx].to_numpy()
gt = lab.loc[test_idx, "user_id"].to_numpy()
draws = RC.participant_draws(gt)
preds = {}
preds["selected_" + best["inputs"] + "_" + best["family"]] = fit_predict(best["family"], INPUTS[best["inputs"]], dev_idx, [test_idx])[0]
preds["frozen_152_dart_dev"] = fit_predict("lightgbm_dart_frozen_config", INPUTS["f_frozen_152"], dev_idx, [test_idx])[0]
preds["phq_only_dart_dev"] = fit_predict("lightgbm_dart_frozen_config", PHQ, dev_idx, [test_idx])[0]

# published 2-phase design, development -> test
dev_p = set(lab.loc[dev_idx, "user_id"])
imp1, clf1 = CROSSFIT.fit_phase1(raw, phase1_cols, raw["ref_label"] & raw["user_id"].isin(dev_p))
cat_o = raw["phq9_cat_end"].copy(); pr_o = pd.DataFrame(0.0, index=raw.index, columns=REPRO.PROBA_COLS)
CROSSFIT.generate_intermediate(raw, phase1_cols, imp1, clf1, ~raw["ref_label"], cat_o, pr_o)
idx = dev_idx.append(test_idx)
X2 = pd.concat([raw.loc[idx, base], REPRO.build_phase2_generated_features(raw, idx, cat_o, pr_o)], axis=1)
imp2 = SimpleImputer(strategy="mean", keep_empty_features=True)
clf2 = REPRO.phase2_dart_factory()
clf2.fit(imp2.fit_transform(X2.loc[dev_idx]), y_all.loc[dev_idx], sample_weight=compute_sample_weight("balanced", y_all.loc[dev_idx]))
preds["published_2phase_dev"] = clf2.predict_proba(imp2.transform(X2.loc[test_idx]))[:, 1]

# saved centralized models from the FL comparison (trained on the 70% train portion)
S = json.loads((RC.ROOT / "scripts/psyche_d_fedavg_vs_fedtree_step1_results.json").read_text())
X4 = json.loads((RC.ROOT / "scripts/psyche_d_treeavg_audit_extend_s4_results.json").read_text())
assert np.array_equal(np.asarray(S["y_test"]), yt) and np.array_equal(np.asarray(X4["y_test"]), yt)
preds["centralized_lightgbm_train70_saved"] = np.asarray(S["lightgbm_dart_same_split_test_predictions"])
preds["centralized_xgboost_train70_saved"] = np.asarray(X4["centralized_xgboost_test_predictions"])

# ---------------- 1.5 / 1.6 model: frozen config fitted on the 70% train portion ----------------
pv, pt = fit_predict("lightgbm_dart_frozen_config", INPUTS["f_frozen_152"], train_idx, [val_idx, test_idx])
preds["frozen_152_dart_train70"] = pt
yv = y_all.loc[val_idx].to_numpy()

T = {k: RC.summarize(yt, p, draws) for k, p in preds.items()}
fro = T["frozen_152_dart_dev"]
pairs = {f"{k} - frozen_152_dart_dev": RC.paired(v, fro) for k, v in T.items() if k != "frozen_152_dart_dev"}
pd.DataFrame([{"model": k, **RC.strip(v)} for k, v in T.items()]).to_csv(RC.OUT / "P1_4_test_models.csv", index=False)
pd.DataFrame([{"comparison": k, **v} for k, v in pairs.items()]).to_csv(RC.OUT / "P1_4_test_paired.csv", index=False)
print(f"1.4 done {time.time() - t0:.0f}s", flush=True)


def at(y, p, thr):
    pr = p >= thr
    tp, fp = int((pr & (y == 1)).sum()), int((pr & (y == 0)).sum())
    tn, fn = int((~pr & (y == 0)).sum()), int((~pr & (y == 1)).sum())
    div = lambda a, b: a / b if b else np.nan  # noqa: E731
    return {"accuracy": (tp + tn) / len(y), "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn)}


def sens_threshold(y, p, target):
    grid = np.round(np.arange(0, 1.0001, 0.0025), 4)
    ok = [t for t in grid if at(y, p, t)["sensitivity"] >= target]
    return float(max(ok))


cand = np.unique(pv)
acc_v = [(at(yv, pv, t)["accuracy"], t) for t in cand]
thr = {"default_0.5": 0.5, "max_accuracy_val": float(max(acc_v)[1]),
       "matched_sensitivity_55.4": sens_threshold(yv, pv, 0.554), "sensitivity_80": sens_threshold(yv, pv, 0.80)}
ops = []
for name, t in thr.items():
    point = at(yt, pt, t)
    boots = [at(yt[ix], pt[ix], t) for ix in draws]
    row = {"operating_point": name, "threshold": t, "val_sensitivity": at(yv, pv, t)["sensitivity"]}
    for m in point:
        row[m] = point[m]
        row[m + "_ci"] = RC.ci([b[m] for b in boots])
    ops.append(row)
maj = float(max(yt.mean(), 1 - yt.mean()))
pd.DataFrame(ops).to_csv(RC.OUT / "P1_5_operating_points_test.csv", index=False)


# ---------------- 1.6 calibration ----------------
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calib(y, p):
    lp = logit(p)
    nll = lambda a: -np.sum(y * (a + lp) - np.log1p(np.exp(a + lp)))  # noqa: E731
    citl = minimize_scalar(nll, bounds=(-10, 10), method="bounded").x
    lr = LogisticRegression(penalty=None, max_iter=1000).fit(lp[:, None], y)
    return {"intercept_citl": float(citl), "slope": float(lr.coef_[0, 0]), "brier": float(np.mean((p - y) ** 2)),
            "mean_predicted": float(p.mean()), "observed_rate": float(y.mean())}


platt = LogisticRegression(penalty=None, max_iter=1000).fit(logit(pv)[:, None], yv)
pt_platt = platt.predict_proba(logit(pt)[:, None])[:, 1]
cal = {}
for name, p in [("frozen_train70_raw", pt), ("frozen_train70_platt_on_validation", pt_platt)]:
    point = calib(yt, p)
    boots = [calib(yt[ix], p[ix]) for ix in draws[:2000]]
    cal[name] = {**point, **{k + "_ci": RC.ci([b[k] for b in boots]) for k in ("intercept_citl", "slope", "brier")},
                 "auroc": float(roc_auc_score(yt, p))}
cal["platt_params"] = {"a": float(platt.intercept_[0]), "b": float(platt.coef_[0, 0])}
pd.DataFrame([{"version": k, **v} for k, v in cal.items() if k != "platt_params"]).to_csv(RC.OUT / "P1_6_calibration.csv", index=False)

fig, ax = plt.subplots(figsize=(4.5, 4.5))
for name, p, mk in [("Uncalibrated", pt, "o"), ("Platt-recalibrated (validation)", pt_platt, "s")]:
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    d = pd.DataFrame({"p": p, "y": yt, "q": q}).groupby("q").mean()
    ax.plot(d["p"], d["y"], marker=mk, label=name)
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Ideal")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("Mean predicted probability (decile)"); ax.set_ylabel("Observed worsening rate")
ax.set_title("PSYCHE-D frozen model, test set (n=1631)", fontsize=9)
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(RC.OUT / "P1_6_calibration_plot.png", dpi=300)

res = {"seeds": {"split": 20260928, "dev_cv": 42, "model": 42, "bootstrap": RC.BOOT_SEED, "n_boot": RC.N_BOOT},
       "split_sizes": {"train": [len(train_idx), int(lab.loc[train_idx, "user_id"].nunique())],
                       "validation": [len(val_idx), int(lab.loc[val_idx, "user_id"].nunique())],
                       "test": [len(test_idx), int(len(set(gt)))]},
       "selection_top5": sel_df.head(5).to_dict("records"), "selected": best.to_dict(),
       "test_models": {k: RC.strip(v) for k, v in T.items()}, "test_paired_vs_frozen_dev": pairs,
       "thresholds": thr, "operating_points": ops, "test_majority_class_rate": maj, "test_positive_rate": float(yt.mean()),
       "calibration": cal, "runtime_sec": round(time.time() - t0)}
(RC.OUT / "P1_holdout_results.json").write_text(json.dumps(res, indent=1, default=float))
print(sel_df.head(8).to_string())
for k, v in T.items():
    print(f"{k:45s} {RC.fmt(v)} AUPRC {v['auprc']:.3f} ({v['auprc_ci'][0]:.3f}-{v['auprc_ci'][1]:.3f})")
for k, v in pairs.items():
    print(f"{k:70s} {v['delta']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]")
print(pd.DataFrame(ops)[["operating_point", "threshold", "accuracy", "sensitivity", "specificity", "ppv", "npv"]].to_string())
print("majority", maj)
print(json.dumps(cal, indent=1))
