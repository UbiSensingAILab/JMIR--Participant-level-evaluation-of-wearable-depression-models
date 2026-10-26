#!/usr/bin/env python3
# REB25-1587_MOD1 research data module; no PII in logs. HRV and PHQ-9 (Baigutanova 2025, CC BY 4.0)
# and RESILIENT (Zenodo 16755408, CC BY 4.0) data stay under gitignored research_datasets/ and scripts/.cache/.
"""JMIR revision, Phase 3 (small datasets). Run with .venv-psyche-d-fl-mlp.

3.1 HRV leakage check with participant-level CIs (5-minute records and segments).
3.2 MOMENT on HRV: reuse the cached frozen MOMENT-1-small embeddings (stage06b) and rerun
    (a) PCA 5/10/20 fitted inside each training fold, (b) L2 logistic regression on all 512
    dimensions with C tuned in the inner grouped loop. Same outer folds as the frozen model.
3.3 Calibration (intercept, calibration-in-the-large, slope, Brier) for the HRV and RESILIENT
    frozen models from pooled outer-fold predictions.

Seeds: C.SEED = 42 for every split, model and bootstrap (np.random.RandomState(42) cluster
bootstrap, 2000 resamples, the same draw for both models in a paired comparison); outer folds
StratifiedGroupKFold(5) with seeds 42..46, inner StratifiedGroupKFold(3) with seed 42+100r+k.
Outputs: results/revision/P3_*.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import hrv_phq9_protocol_common as C  # noqa: E402

OUT = ROOT / "results" / "revision"
OUT.mkdir(parents=True, exist_ok=True)
R = {"seeds": {"split_model_bootstrap": C.SEED, "outer_fold_seeds": [C.SEED + r for r in range(C.OUTER_REPEATS)],
               "n_boot": C.N_BOOT, "bootstrap": "participant cluster bootstrap, np.random.RandomState(42)"}}


def r4(x):
    return None if x is None else round(float(x), 4)


# ------------------------------------------------------------------ 3.1 leakage check
def task_3_1(seg):
    s = C.calendar_rows(C.load_sensor_rows())
    s = s.merge(seg[["deviceId", "segment", "y"]], on=["deviceId", "segment"], how="inner")
    X = s[C.SENSOR_CHANNELS].to_numpy(dtype=np.float64)
    X[~np.isfinite(X)] = np.nan
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)  # as in stage02 (column means; unsupervised)
    y, g = s["y"].to_numpy(), s["deviceId"].to_numpy()
    rows = []
    preds = {}
    for name, cv in [("record_split_random_row_kfold", KFold(5, shuffle=True, random_state=C.SEED)),
                     ("participant_split_grouped", StratifiedGroupKFold(5, shuffle=True, random_state=C.SEED))]:
        p = np.zeros(len(y))
        for tr, te in cv.split(X, y, g):
            m = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, class_weight="balanced", random_state=C.SEED, n_jobs=-1)
            m.fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        preds[name] = p
        ci, nb = C.boot_ci(y, p, g, "auroc")
        rows.append({"level": "5-minute records", "split": name, "n_rows": len(y), "n_participants": int(len(np.unique(g))),
                     "auroc": r4(roc_auc_score(y, p)), "ci_low": r4(ci[0]), "ci_high": r4(ci[1]), "n_valid_resamples": nb})
    d = C.paired_boot_diff(y, preds["record_split_random_row_kfold"], preds["participant_split_grouped"], g)
    rows.append({"level": "5-minute records", "split": "difference record - participant", "n_rows": len(y),
                 "n_participants": int(len(np.unique(g))), "auroc": r4(d["diff"]), "ci_low": r4(d["ci"][0]),
                 "ci_high": r4(d["ci"][1]), "n_valid_resamples": d["n_valid_resamples"]})

    # segment level: RF depth 4 on the SD_sensor_diurnal arm, 5 repeats (as stage02)
    Xs, ys, gs = C.Xy(seg, C.arm_columns(seg, "SD_sensor_diurnal"))
    base = clone(C.model_specs()["RF"][0]).set_params(m__max_depth=4, m__min_samples_leaf=3)
    for name, cvc in [("record_split_random_segment_kfold", StratifiedKFold), ("participant_split_grouped", StratifiedGroupKFold)]:
        P = np.zeros((len(ys), 5))
        for rep in range(5):
            cv = cvc(5, shuffle=True, random_state=C.SEED + rep)
            for tr, te in cv.split(Xs, ys, gs):
                e = clone(base)
                e.fit(Xs[tr], ys[tr])
                P[te, rep] = e.predict_proba(Xs[te])[:, 1]
        mean_rep = float(np.mean([roc_auc_score(ys, P[:, r]) for r in range(5)]))
        pa = P.mean(1)
        ci, nb = C.boot_ci(ys, pa, gs, "auroc")
        rows.append({"level": "segments (14 days)", "split": name, "n_rows": len(ys), "n_participants": int(len(np.unique(gs))),
                     "auroc": r4(roc_auc_score(ys, pa)), "auroc_mean_of_5_repeats": r4(mean_rep),
                     "ci_low": r4(ci[0]), "ci_high": r4(ci[1]), "n_valid_resamples": nb})
    pd.DataFrame(rows).to_csv(OUT / "P3_1_hrv_leakage_participant_ci.csv", index=False)
    R["3.1"] = rows
    print(pd.DataFrame(rows).to_string(), flush=True)


# ------------------------------------------------------------------ 3.2 MOMENT on HRV
LR_GRID = [{"m__C": c} for c in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]]
L2_GRID = [{"m__C": c} for c in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]]


def pipe(n_emb, n_other, n_pc):
    emb = [("imp", SimpleImputer()), ("sc", StandardScaler())]
    if n_pc:
        emb.append(("pca", PCA(n_pc, random_state=C.SEED)))
    parts = [("emb", Pipeline(emb), list(range(n_emb)))]
    if n_other:
        parts.append(("oth", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]), list(range(n_emb, n_emb + n_other))))
    return Pipeline([("ct", ColumnTransformer(parts)),
                     ("m", LogisticRegression(penalty="l2", class_weight="balanced", max_iter=5000, solver="liblinear"))])


def task_3_2(seg):
    E = np.load(C.CACHE / "hrv_phq9_protocol_moment_embeddings.npz", allow_pickle=True)
    assert (E["device"] == seg["deviceId"].to_numpy()).all() and (E["segment"] == seg["segment"].to_numpy()).all()
    y, g = seg["y"].to_numpy(), seg["deviceId"].to_numpy()
    pf = np.nanmean(np.load(C.CACHE / "hrv_phq9_protocol_frozen_oof.npz")["oof"], axis=1)
    hr = E["hr_mean"]
    phq, _, _ = C.Xy(seg, ["PHQ9_1"])
    cnt = E["hr_count"]
    win = {"windows_total": int(cnt.sum()), "segments": int(len(cnt)), "windows_per_segment_min": int(cnt.min()),
           "windows_per_segment_median": float(np.median(cnt)), "windows_per_segment_max": int(cnt.max()),
           "embedding_dim_per_window": int(hr.shape[1]), "segment_vector_used_by_classifier": "mean over windows (512); the SD vector is cached but not used"}
    frozen = {"auroc": r4(roc_auc_score(y, pf)), "ci": [r4(v) for v in C.boot_ci(y, pf, g)[0]]}
    configs = [("PCA16 (original, reproduction check)", 16, LR_GRID)] + [(f"PCA{k} in fold", k, LR_GRID) for k in (5, 10, 20)] + \
              [("L2 LR on all 512 dims, C tuned inner loop", 0, L2_GRID)]
    rows = []
    for label, npc, grid in configs:
        for inputs, X, no in [("embeddings alone", hr, 0), ("embeddings + day-0 PHQ-9", np.hstack([hr, phq]), 1)]:
            res = C.nested_cv(pipe(hr.shape[1], no, npc), grid, X, y, g)
            p = np.nanmean(res["oof"], axis=1)
            ci, _ = C.boot_ci(y, p, g, "auroc")
            d = C.paired_boot_diff(y, p, pf, g)
            chosen = pd.Series([c["params"]["m__C"] for c in res["chosen"]]).value_counts().to_dict()
            rows.append({"config": label, "inputs": inputs, "auroc": r4(roc_auc_score(y, p)), "ci_low": r4(ci[0]), "ci_high": r4(ci[1]),
                         "mean_auroc_over_repeats": r4(res["mean_auroc"]), "diff_vs_frozen": r4(d["diff"]),
                         "diff_ci_low": r4(d["ci"][0]), "diff_ci_high": r4(d["ci"][1]), "verdict": d["verdict"],
                         "C_chosen_counts": json.dumps({str(k): int(v) for k, v in chosen.items()})})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "P3_2_hrv_moment_rerun.csv", index=False)
    R["3.2"] = {"windowing": win, "frozen_reference": frozen, "rows": rows}


# ------------------------------------------------------------------ 3.3 calibration
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def citl(y, lp):
    """Intercept a in logit P(y=1) = a + lp (slope fixed at 1), by Newton's method."""
    a = 0.0
    for _ in range(100):
        mu = 1 / (1 + np.exp(-(a + lp)))
        step = np.sum(y - mu) / np.sum(mu * (1 - mu))
        a += step
        if abs(step) < 1e-10:
            break
    return a


def cal(y, p):
    lp = logit(p)
    m = LogisticRegression(penalty=None, max_iter=10000).fit(lp[:, None], y)
    return {"intercept_citl": citl(y, lp), "slope": float(m.coef_[0, 0]), "intercept_with_slope": float(m.intercept_[0]),
            "brier": float(brier_score_loss(y, p)), "brier_null_prevalence": float(np.mean((y - y.mean()) ** 2)),
            "mean_predicted": float(p.mean()), "observed_rate": float(y.mean())}


def cal_boot(y, p, g):
    uniq, pos = C._cluster_index(g)
    rng = np.random.RandomState(C.SEED)
    vals = []
    for _ in range(C.N_BOOT):
        ix = np.concatenate([pos[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[ix])) < 2:
            continue
        c = cal(y[ix], p[ix])
        vals.append([c["intercept_citl"], c["slope"], c["brier"]])
    v = np.array(vals)
    return {k: [float(np.percentile(v[:, i], 2.5)), float(np.percentile(v[:, i], 97.5))] for i, k in enumerate(["intercept_citl", "slope", "brier"])}, len(v)


def task_3_3(seg):
    y, g = seg["y"].to_numpy(), seg["deviceId"].to_numpy()
    pf = np.nanmean(np.load(C.CACHE / "hrv_phq9_protocol_frozen_oof.npz")["oof"], axis=1)
    rj = json.loads((ROOT / "scripts" / "resilient_protocol_oof_predictions.json").read_text())
    yr = np.asarray(rj["y_phq9_ge10"])
    pr = np.asarray(rj["stage4_arm_oof_repeat_avg"]["sensor_demo (candidate)"])
    gr = np.arange(len(yr))  # one row per participant
    rows = []
    for name, yy, pp, gg in [("HRV and PHQ-9 frozen (day-0 PHQ-9, LR C=0.001, balanced)", y, pf, g),
                             ("RESILIENT frozen (RF sensors + demographics)", yr, pr, gr)]:
        c = cal(yy, pp)
        ci, nb = cal_boot(yy, pp, gg)
        rows.append({"model": name, "n_rows": len(yy), "n_participants": int(len(np.unique(gg))), "n_positive": int(yy.sum()),
                     "auroc_check": r4(roc_auc_score(yy, pp)),
                     "calibration_in_the_large": r4(c["intercept_citl"]), "citl_ci": "%.3f to %.3f" % tuple(ci["intercept_citl"]),
                     "slope": r4(c["slope"]), "slope_ci": "%.3f to %.3f" % tuple(ci["slope"]),
                     "intercept_with_slope": r4(c["intercept_with_slope"]),
                     "brier": r4(c["brier"]), "brier_ci": "%.3f to %.3f" % tuple(ci["brier"]),
                     "brier_null_prevalence": r4(c["brier_null_prevalence"]),
                     "mean_predicted": r4(c["mean_predicted"]), "observed_rate": r4(c["observed_rate"]),
                     "prediction_range": "%.3f-%.3f" % (pp.min(), pp.max()), "n_valid_resamples": nb})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "P3_3_calibration_small_datasets.csv", index=False)
    R["3.3"] = rows


if __name__ == "__main__":
    seg = C.build_segments()
    task_3_3(seg)
    task_3_2(seg)
    task_3_1(seg)
    (OUT / "P3_results.json").write_text(json.dumps(R, indent=2, default=str))
