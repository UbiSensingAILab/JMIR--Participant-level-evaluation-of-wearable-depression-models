#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. `research_datasets/` is gitignored.
"""Plotting-only rerun of final_outcome_calib.py task T9: refits the same 3 models on the same
70/15/15 split, asserts the calibration numbers equal the saved T9_calibration.csv, and saves the
decile points (T9_calibration_deciles.csv) so Figure 3 can be redrawn. No new analysis."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
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
lab = raw[raw["ref_label"]].copy()
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
ARMS = {"P": P, "P+S": P + S, "P+S+L": P + S + L, "P+S+W": P + S + W, "P+S+L+W (frozen)": base + P,
        "R3 forecast: P+S+SM1/SM2 (W+L)": P + S + SMW + SML}
g = lab["user_id"].to_numpy()
y_primary = ((lab["phq9_cat_end"] - lab["phq9_cat_start"]) > 0).astype(int).to_numpy()
y5 = ((lab["phq9_score_end"] - lab["phq9_score_start"]) >= 5).astype(int).to_numpy()
n = len(y5)


def fit_pred(cols, tr, te_list, y):
    imp = SimpleImputer(strategy="mean", keep_empty_features=True)
    clf = REPRO.phase2_dart_factory()
    Xtr = imp.fit_transform(X.iloc[tr][cols].to_numpy(float)) if isinstance(tr, np.ndarray) else imp.fit_transform(X.loc[tr, cols].to_numpy(float))
    ytr = y[tr] if isinstance(tr, np.ndarray) else y.loc[tr].to_numpy()
    clf.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    return [clf.predict_proba(imp.transform((X.iloc[t] if isinstance(t, np.ndarray) else X.loc[t])[cols].to_numpy(float)))[:, 1] for t in te_list]


# ================= T9 =================
cache, labeled, fcols, yser, train_idx, val_idx, test_idx = load_split_and_frame()
y_all = pd.Series(y_primary, index=lab.index)
assert (yser.loc[lab.index].to_numpy() == y_primary).all()
yv, yt = y_all.loc[val_idx].to_numpy(), y_all.loc[test_idx].to_numpy()
draws_t = RC.participant_draws(lab.loc[test_idx, "user_id"].to_numpy())


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calib(y, p):
    lp = logit(p)
    citl = minimize_scalar(lambda a: -np.sum(y * (a + lp) - np.log1p(np.exp(a + lp))), bounds=(-10, 10), method="bounded").x
    lr = LogisticRegression(penalty=None, max_iter=1000).fit(lp[:, None], y)
    return {"intercept_citl": float(citl), "slope": float(lr.coef_[0, 0]), "brier": float(np.mean((p - y) ** 2)),
            "mean_predicted": float(p.mean()), "observed_rate": float(y.mean())}



CAL_MODELS = {"Frozen (P+S+L+W, 152)": "P+S+L+W (frozen)", "R3 forecast": "R3 forecast: P+S+SM1/SM2 (W+L)", "Development-selected (P+S+L, 41)": "P+S+L"}
saved = pd.read_csv(RC.OUT / "T9_calibration.csv")
rows, check = [], []
for label, arm in CAL_MODELS.items():
    pv, pt = fit_pred(ARMS[arm], train_idx, [val_idx, test_idx], y_all)
    platt = LogisticRegression(penalty=None, max_iter=1000).fit(logit(pv)[:, None], yv)
    pt_platt = platt.predict_proba(logit(pt)[:, None])[:, 1]
    for ver, p in [("uncalibrated", pt), ("Platt (validation)", pt_platt)]:
        c = calib(yt, p)
        s = saved[(saved.model == label) & (saved.version == ver)].iloc[0]
        check.append({"model": label, "version": ver, **{k: c[k] - float(s[k]) for k in ("intercept_citl", "slope", "brier")}})
        d = pd.DataFrame({"p": p, "y": yt, "q": pd.qcut(p, 10, labels=False, duplicates="drop")}).groupby("q").mean()
        for q, r in d.iterrows():
            rows.append({"model": label, "version": ver, "decile": int(q), "mean_predicted": r["p"], "observed_rate": r["y"],
                         "citl": c["intercept_citl"], "slope": c["slope"], "brier": c["brier"]})
    print(label, f"{time.time() - t0:.0f}s", flush=True)
ck = pd.DataFrame(check)
print(ck.to_string())
assert ck[["intercept_citl", "slope", "brier"]].abs().max().max() < 1e-9, "refit does not reproduce saved T9 calibration"
pd.DataFrame(rows).to_csv(RC.OUT / "T9_calibration_deciles.csv", index=False)
print("reproduced exactly; deciles saved")
