#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. `research_datasets/` is gitignored.
"""Updated analysis plan, Tier 1, R6 (held-out test and operating points) and R7
(calibration of the frozen and R3 forecasting models) on the fixed 70/15/15 split.

R6  development = train + validation (85%). Input sets (R2 + R3) x model families
    (LightGBM DART frozen config, LightGBM GBDT, XGBoost, LR), pooled OOF AUROC from
    StratifiedGroupKFold(5, shuffle, 42) on development only; best refitted on development
    and scored once on test, together with the frozen configuration.
    Operating points: frozen configuration fitted on the 70% training portion, thresholds
    chosen on validation, scored once on test.
R7  frozen and R3 forecasting models fitted on the 70% training portion; calibration on
    test; Platt scaling fitted on validation. 10-bin (deciles of predicted risk) plot.
Participant bootstrap on the 606 test participants, 2000 draws, seed 42.
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
import rev_common as RC  # noqa: E402
from psyche_d_fedavg_vs_fedtree_step4_fed_xgb_bagging import load_split_and_frame  # noqa: E402

t0 = time.time()
raw = REPRO.load_raw()
base = [c for c in raw.columns if c not in REPRO.LEAKAGE_COLS and c not in ("user_id", "user_mth", "ref_label")]
fmap = pd.read_csv(RC.OUT / "P0_4_psyche_d_feature_groups.csv").set_index("feature")["group"]
P = ["phq9_score_start", "phq9_cat_start"]
S = [c for c in base if fmap[c] == "screener"]
L = [c for c in base if fmap[c] == "lifestyle_medication"]
W = [c for c in base if fmap[c] == "wearable"]
lab = raw[raw["ref_label"]]
month = raw.set_index(["user_id", "user_mth"])[W + L]
blocks = []
for tag, off in (("sm1", 2), ("sm2", 1)):
    blk = month.reindex(list(zip(lab["user_id"], lab["user_mth"] - off)))
    blk.index = lab.index
    blk.columns = [f"{c}__{tag}" for c in blk.columns]
    blocks.append(blk)
X = pd.concat([lab[base + P]] + blocks, axis=1)
SMW = [f"{c}__{t}" for t in ("sm1", "sm2") for c in W]
SML = [f"{c}__{t}" for t in ("sm1", "sm2") for c in L]
y_all = ((lab["phq9_cat_end"] - lab["phq9_cat_start"]) > 0).astype(int)
INPUTS = {"P": P, "S": S, "L": L, "W": W, "P+S": P + S, "P+S+L": P + S + L, "P+S+W": P + S + W,
          "P+S+L+W (frozen)": base + P, "R3 forecast (W+L)": P + S + SMW + SML, "R3 forecast Fitbit only": P + S + SMW}
FAMILIES = {
    "lightgbm_dart_frozen_config": lambda: REPRO.phase2_dart_factory(),
    "lightgbm_gbdt": lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=42, verbosity=-1, n_jobs=-1),
    "xgboost": lambda: xgb.XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                                         tree_method="hist", random_state=42, n_jobs=-1),
    "logistic_regression": lambda: Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=3000, random_state=42))]),
}
cache, labeled, fcols, yser, train_idx, val_idx, test_idx = load_split_and_frame()
dev_idx = train_idx.append(val_idx)


def fit_predict(fam, cols, tr, te_list):
    imp = SimpleImputer(strategy="mean", keep_empty_features=True)
    clf = FAMILIES[fam]()
    sw = compute_sample_weight("balanced", y_all.loc[tr])
    Xtr = imp.fit_transform(X.loc[tr, cols].to_numpy(float))
    if fam == "logistic_regression":
        clf.fit(Xtr, y_all.loc[tr], lr__sample_weight=sw)
    else:
        clf.fit(Xtr, y_all.loc[tr], sample_weight=sw)
    return [clf.predict_proba(imp.transform(X.loc[t, cols].to_numpy(float)))[:, 1] for t in te_list]


# ---------------- R6 selection on development ----------------
gd, yd = lab.loc[dev_idx, "user_id"].to_numpy(), y_all.loc[dev_idx].to_numpy()
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.zeros(len(yd)), yd, gd))
sel = []
for iname, cols in INPUTS.items():
    for fam in FAMILIES:
        oof = np.full(len(yd), np.nan)
        for tr, te in folds:
            oof[te] = fit_predict(fam, cols, dev_idx[tr], [dev_idx[te]])[0]
        sel.append({"inputs": iname, "n_inputs": len(cols), "family": fam, "dev_cv_auroc": float(roc_auc_score(yd, oof))})
    print(f"selection {iname} {time.time() - t0:.0f}s", flush=True)
sel_df = pd.DataFrame(sel).sort_values("dev_cv_auroc", ascending=False)
best = sel_df.iloc[0]
yt = y_all.loc[test_idx].to_numpy()
gt = lab.loc[test_idx, "user_id"].to_numpy()
draws = RC.participant_draws(gt)
T = {f"selected: {best['inputs']} / {best['family']}": fit_predict(best["family"], INPUTS[best["inputs"]], dev_idx, [test_idx])[0]}
for k in ["P+S+L+W (frozen)", "P", "P+S", "P+S+L", "R3 forecast (W+L)"]:
    T[f"{k} / DART (development-trained)"] = fit_predict("lightgbm_dart_frozen_config", INPUTS[k], dev_idx, [test_idx])[0]
TS = {k: RC.summarize(yt, p, draws) for k, p in T.items()}
ref = TS["P+S+L+W (frozen) / DART (development-trained)"]
TP = [{"comparison": f"{k} - frozen", **RC.paired(v, ref)} for k, v in TS.items() if v is not ref]


# ---------------- R6 operating points + R7 calibration (70% training fit) ----------------
def at(y, p, thr):
    pr = p >= thr
    tp, fp = int((pr & (y == 1)).sum()), int((pr & (y == 0)).sum())
    tn, fn = int((~pr & (y == 0)).sum()), int((~pr & (y == 1)).sum())
    dv = lambda a, b: a / b if b else np.nan  # noqa: E731
    return {"accuracy": (tp + tn) / len(y), "sensitivity": dv(tp, tp + fn), "specificity": dv(tn, tn + fp), "ppv": dv(tp, tp + fp), "npv": dv(tn, tn + fn)}


def sens_thr(y, p, target):
    grid = np.round(np.arange(0, 1.0001, 0.0025), 4)
    return float(max(t for t in grid if at(y, p, t)["sensitivity"] >= target))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calib(y, p):
    lp = logit(p)
    citl = minimize_scalar(lambda a: -np.sum(y * (a + lp) - np.log1p(np.exp(a + lp))), bounds=(-10, 10), method="bounded").x
    lr = LogisticRegression(penalty=None, max_iter=1000).fit(lp[:, None], y)
    return {"intercept_citl": float(citl), "slope": float(lr.coef_[0, 0]), "brier": float(np.mean((p - y) ** 2)),
            "mean_predicted": float(p.mean()), "observed_rate": float(y.mean())}


yv = y_all.loc[val_idx].to_numpy()
ops, cal, curves = [], {}, {}
for mname in ["P+S+L+W (frozen)", "R3 forecast (W+L)"]:
    pv, pt = fit_predict("lightgbm_dart_frozen_config", INPUTS[mname], train_idx, [val_idx, test_idx])
    if mname.startswith("P+S+L+W"):
        thr = {"default 0.5": 0.5, "maximum accuracy (validation)": float(max((at(yv, pv, t)["accuracy"], t) for t in np.unique(pv))[1]),
               "matched sensitivity 55.4%": sens_thr(yv, pv, 0.554), "80% sensitivity": sens_thr(yv, pv, 0.80)}
        for name, t in thr.items():
            pt_ = at(yt, pt, t)
            bs = [at(yt[ix], pt[ix], t) for ix in draws]
            ops.append({"operating_point": name, "threshold": t, **{m: pt_[m] for m in pt_}, **{m + "_ci": RC.ci([b[m] for b in bs]) for m in pt_}})
    platt = LogisticRegression(penalty=None, max_iter=1000).fit(logit(pv)[:, None], yv)
    pt_platt = platt.predict_proba(logit(pt)[:, None])[:, 1]
    for ver, p in [("uncalibrated", pt), ("Platt (validation)", pt_platt)]:
        point = calib(yt, p)
        bs = [calib(yt[ix], p[ix]) for ix in draws]
        cal[f"{mname} | {ver}"] = {**point, **{k + "_ci": RC.ci([b[k] for b in bs]) for k in ("intercept_citl", "slope", "brier")},
                                   "auroc": float(roc_auc_score(yt, p))}
        curves[f"{mname} | {ver}"] = p
    cal[f"{mname} | platt_params"] = {"a": float(platt.intercept_[0]), "b": float(platt.coef_[0, 0])}

fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), sharey=True)
for ax, mname in zip(axes, ["P+S+L+W (frozen)", "R3 forecast (W+L)"]):
    for ver, mk in [("uncalibrated", "o"), ("Platt (validation)", "s")]:
        p = curves[f"{mname} | {ver}"]
        d = pd.DataFrame({"p": p, "y": yt, "q": pd.qcut(p, 10, labels=False, duplicates="drop")}).groupby("q").mean()
        ax.plot(d["p"], d["y"], marker=mk, label=ver)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"{'Frozen model' if 'frozen' in mname else 'Forecasting model (R3)'}, test n=1631", fontsize=9)
    ax.set_xlabel("Mean predicted probability (decile)", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
axes[0].set_ylabel("Observed worsening rate", fontsize=8)
fig.tight_layout()
fig.savefig(RC.OUT / "R7_calibration_frozen_and_forecast.png", dpi=300)

maj = float(max(yt.mean(), 1 - yt.mean()))
sel_df.to_csv(RC.OUT / "R6_dev_cv_selection.csv", index=False)
pd.DataFrame([{"model": k, **RC.strip(v)} for k, v in TS.items()]).to_csv(RC.OUT / "R6_test_models.csv", index=False)
pd.DataFrame(TP).to_csv(RC.OUT / "R6_test_paired_vs_frozen.csv", index=False)
pd.DataFrame(ops).to_csv(RC.OUT / "R6_operating_points_test.csv", index=False)
pd.DataFrame([{"model_version": k, **v} for k, v in cal.items() if "platt_params" not in k]).to_csv(RC.OUT / "R7_calibration_psyche_d.csv", index=False)
res = {"seeds": {"split": 20260928, "dev_cv": 42, "model": 42, "bootstrap": RC.BOOT_SEED, "n_boot": RC.N_BOOT},
       "selection_top8": sel_df.head(8).to_dict("records"), "selected": best.to_dict(), "test": {k: RC.strip(v) for k, v in TS.items()},
       "test_paired_vs_frozen": TP, "operating_points": ops, "test_majority_class_rate": maj, "calibration": cal,
       "small_dataset_calibration_source": "results/revision/P3_3_calibration_small_datasets.csv", "runtime_sec": round(time.time() - t0)}
(RC.OUT / "R6_R7_results.json").write_text(json.dumps(res, indent=1, default=float))
print(sel_df.head(10).to_string())
for k, v in TS.items():
    print(f"{k:55s} {RC.fmt(v)} AUPRC {v['auprc']:.3f}")
for r in TP:
    print(f"{r['comparison']:65s} {r['delta']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]")
print(pd.DataFrame(ops)[["operating_point", "threshold", "accuracy", "sensitivity", "specificity", "ppv", "npv"]].round(3).to_string())
print("majority", round(maj, 4))
print(json.dumps(cal, indent=0, default=float))
