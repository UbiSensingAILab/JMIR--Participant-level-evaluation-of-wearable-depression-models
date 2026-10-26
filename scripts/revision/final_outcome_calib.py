#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. `research_datasets/` is gitignored.
"""Final analysis run, task 6 (outcome sensitivity) and task 9 (calibration of the
development-selected P+S+L model).

T6  Second outcome: PHQ-9 score increase of >= 5 points from start to end of the period
    (phq9_score_end - phq9_score_start >= 5). Arms P, P+S, P+S+L, P+S+W, frozen (152) and
    the R3 forecasting model (P + S + month-1/month-2 W and L), frozen model family
    (LightGBM DART), mean imputation, balanced weights. Folds: the SAME participant-grouped
    folds as the primary analysis (StratifiedGroupKFold(5, shuffle, 42) generated from the
    primary outcome), so only the label changes. Participant bootstrap, 2000 draws, seed 42.
T9  Fixed 70/15/15 split (seed 20260928). Frozen, R3 forecast and P+S+L DART models fitted
    on the 70% training portion (as R7), calibration on the test set before and after Platt
    scaling fitted on validation; 10-bin (deciles of predicted risk) plot. Test-set
    operating points for P+S+L chosen on validation exactly as R6 does for the frozen model.
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


# ================= T6 =================
skf = StratifiedGroupKFold(n_splits=REPRO.N_SPLITS, shuffle=True, random_state=REPRO.RANDOM_STATE)
folds = list(skf.split(np.zeros(n), y_primary, g))
oof = {k: np.full(n, np.nan) for k in ARMS}
for f, (tr, te) in enumerate(folds):
    for k, cols in ARMS.items():
        oof[k][te] = fit_pred(cols, tr, [te], y5)[0]
    print(f"T6 fold {f + 1}/5 {time.time() - t0:.0f}s", flush=True)
# check: primary-outcome frozen OOF with these folds equals the saved Tier 1 OOF
prev = pd.read_csv(RC.CACHE / "R_tier1_cv_oof.csv")
assert (prev["row_id"].astype(str).to_numpy() == lab.index.astype(str).to_numpy()).all()
fold_check = np.full(n, np.nan)
for tr, te in folds[:1]:
    fold_check[te] = fit_pred(ARMS["P+S+L+W (frozen)"], tr, [te], y_primary)[0]
m0 = ~np.isnan(fold_check)
fold_repro_maxdiff = float(np.max(np.abs(fold_check[m0] - prev["P+S+L+W (frozen)"].to_numpy()[m0])))

pd.DataFrame({"user_id": g, "row_id": lab.index, "y_phq9_increase_ge5": y5, "y_primary": y_primary, **oof}).to_csv(RC.CACHE / "T6_outcome_ge5_oof.csv", index=False)
draws = RC.participant_draws(g)
Sm = {k: RC.summarize(y5, p, draws) for k, p in oof.items()}
pairs = [("(P+S) - P", "P+S", "P"), ("(P+S+W) - (P+S)", "P+S+W", "P+S"), ("(P+S+L) - (P+S)", "P+S+L", "P+S"),
         ("frozen - (P+S+L)", "P+S+L+W (frozen)", "P+S+L"), ("frozen - P", "P+S+L+W (frozen)", "P"),
         ("R3 forecast - (P+S)", "R3 forecast: P+S+SM1/SM2 (W+L)", "P+S")]
PR6 = [{"comparison": nm, **RC.paired(Sm[a], Sm[b])} for nm, a, b in pairs]
prim = pd.read_csv(RC.OUT / "R2_R3_R4_paired_differences.csv")
t6_meta = {"outcome": "phq9_score_end - phq9_score_start >= 5", "n_samples": n, "n_participants": int(len(set(g))),
           "n_positive": int(y5.sum()), "prevalence": float(y5.mean()), "n_positive_participants": int(len(set(g[y5 == 1]))),
           "overlap_with_primary": {"both": int(((y5 == 1) & (y_primary == 1)).sum()), "ge5_only": int(((y5 == 1) & (y_primary == 0)).sum()),
                                    "primary_only": int(((y5 == 0) & (y_primary == 1)).sum())},
           "folds": "identical to primary analysis (StratifiedGroupKFold(5, shuffle, 42) on the primary outcome)",
           "fold_reproduction_max_abs_diff_fold1": fold_repro_maxdiff}
pd.DataFrame([{"arm": k, "n_inputs": len(ARMS[k]), **RC.strip(v)} for k, v in Sm.items()]).to_csv(RC.OUT / "T6_outcome_ge5_arms.csv", index=False)
pd.DataFrame(PR6).to_csv(RC.OUT / "T6_outcome_ge5_paired.csv", index=False)
print(json.dumps(t6_meta, indent=1))
for k, v in Sm.items():
    print(f"T6 {k:40s} {RC.fmt(v)} AUPRC {v['auprc']:.3f}")
for r in PR6:
    print(f"T6 {r['comparison']:25s} {r['delta']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]")


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


def at(y, p, thr):
    pr = p >= thr
    tp, fp = int((pr & (y == 1)).sum()), int((pr & (y == 0)).sum())
    tn, fn = int((~pr & (y == 0)).sum()), int((~pr & (y == 1)).sum())
    dv = lambda a, b: a / b if b else np.nan  # noqa: E731
    return {"accuracy": (tp + tn) / len(y), "sensitivity": dv(tp, tp + fn), "specificity": dv(tn, tn + fp), "ppv": dv(tp, tp + fp), "npv": dv(tn, tn + fn)}


def sens_thr(y, p, target):
    grid = np.round(np.arange(0, 1.0001, 0.0025), 4)
    return float(max(t for t in grid if at(y, p, t)["sensitivity"] >= target))


CAL_MODELS = {"Frozen (P+S+L+W, 152)": "P+S+L+W (frozen)", "R3 forecast": "R3 forecast: P+S+SM1/SM2 (W+L)", "Development-selected (P+S+L, 41)": "P+S+L"}
cal, curves, ops = {}, {}, []
for label, arm in CAL_MODELS.items():
    pv, pt = fit_pred(ARMS[arm], train_idx, [val_idx, test_idx], y_all)
    platt = LogisticRegression(penalty=None, max_iter=1000).fit(logit(pv)[:, None], yv)
    pt_platt = platt.predict_proba(logit(pt)[:, None])[:, 1]
    pv_platt = platt.predict_proba(logit(pv)[:, None])[:, 1]
    for ver, p in [("uncalibrated", pt), ("Platt (validation)", pt_platt)]:
        point = calib(yt, p)
        bs = [calib(yt[ix], p[ix]) for ix in draws_t]
        cal[(label, ver)] = {**point, **{k + "_ci": RC.ci([b[k] for b in bs]) for k in ("intercept_citl", "slope", "brier")},
                             "auroc": float(roc_auc_score(yt, p)), "platt_a": float(platt.intercept_[0]), "platt_b": float(platt.coef_[0, 0])}
        curves[(label, ver)] = p
    if arm in ("P+S+L", "P+S+L+W (frozen)"):
        for ver, v_, t_ in [("uncalibrated", pv, pt), ("Platt (validation)", pv_platt, pt_platt)]:
            thr = {"default 0.5": 0.5, "maximum accuracy (validation)": float(max((at(yv, v_, t)["accuracy"], t) for t in np.unique(v_))[1]),
                   "matched sensitivity 55.4%": sens_thr(yv, v_, 0.554), "80% sensitivity": sens_thr(yv, v_, 0.80)}
            for name, t in thr.items():
                r = at(yt, t_, t)
                bs = [at(yt[ix], t_[ix], t) for ix in draws_t]
                ops.append({"model": label, "scores": ver, "operating_point": name, "threshold": t, **r,
                            **{m + "_ci": RC.ci([b[m] for b in bs]) for m in r}})
    print(f"T9 {label} {time.time() - t0:.0f}s", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3), sharey=True)
for ax, label in zip(axes, CAL_MODELS):
    for ver, mk in [("uncalibrated", "o"), ("Platt (validation)", "s")]:
        p = curves[(label, ver)]
        d = pd.DataFrame({"p": p, "y": yt, "q": pd.qcut(p, 10, labels=False, duplicates="drop")}).groupby("q").mean()
        c = cal[(label, ver)]
        ax.plot(d["p"], d["y"], marker=mk, label=f"{ver}: CITL {c['intercept_citl']:.2f}, slope {c['slope']:.2f}, Brier {c['brier']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"{label}\ntest n=1631", fontsize=9)
    ax.set_xlabel("Mean predicted probability (decile)", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper left")
axes[0].set_ylabel("Observed worsening rate", fontsize=8)
fig.tight_layout()
fig.savefig(RC.OUT / "T9_calibration_frozen_forecast_selected.png", dpi=300)

r7 = pd.read_csv(RC.OUT / "R7_calibration_psyche_d.csv")
repro = []
for (label, ver), v in cal.items():
    key = {"Frozen (P+S+L+W, 152)": "P+S+L+W (frozen)", "R3 forecast": "R3 forecast (W+L)"}.get(label)
    if key:
        old = r7[r7.model_version == f"{key} | {ver}"].iloc[0]
        repro.append({"model": label, "version": ver, "brier_now": v["brier"], "brier_R7": float(old["brier"]), "slope_now": v["slope"], "slope_R7": float(old["slope"])})
pd.DataFrame([{"model": k[0], "version": k[1], **v} for k, v in cal.items()]).to_csv(RC.OUT / "T9_calibration.csv", index=False)
pd.DataFrame(ops).to_csv(RC.OUT / "T9_operating_points_test.csv", index=False)
pd.DataFrame(repro).to_csv(RC.OUT / "T9_reproduction_check_vs_R7.csv", index=False)
(RC.OUT / "T6_T9_results.json").write_text(json.dumps({
    "T6": {"meta": t6_meta, "arms": {k: RC.strip(v) for k, v in Sm.items()}, "paired": PR6},
    "T9": {"calibration": {f"{k[0]} | {k[1]}": v for k, v in cal.items()}, "operating_points": ops, "repro_vs_R7": repro,
           "test_n": int(len(yt)), "test_prevalence": float(yt.mean())},
    "runtime_sec": round(time.time() - t0)}, indent=1, default=float))
print(pd.DataFrame([{"model": k[0], "version": k[1], **{kk: vv for kk, vv in v.items() if not kk.endswith("_ci")}} for k, v in cal.items()]).round(4).to_string())
print(pd.DataFrame(repro).to_string())
print(pd.DataFrame(ops)[["model", "scores", "operating_point", "threshold", "accuracy", "sensitivity", "specificity", "ppv", "npv"]].round(3).to_string())
