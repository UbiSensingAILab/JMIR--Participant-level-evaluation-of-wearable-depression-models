#!/usr/bin/env python3
# REB25-1587_MOD1 research data module; no PII in logs (device IDs are pseudonymous codes)
"""Shared building blocks for the HRV + PHQ-9 (Ref 2) dataset study protocol run
(src/data/datasetStudyProtocol.md). Every hrv_phq9_protocol_stage*.py script imports this.

Unit of analysis: one participant-segment. Each participant contributes up to two
segments on the shared study calendar (sleep-diary dates 2021-03-09 .. 2021-04-05, KST):
  segment A = study days 1-14  -> label from PHQ9_2 (mid-study, covers the last 2 weeks)
  segment B = study days 15-28 -> label from PHQ9_F (end of study, covers the last 2 weeks)
Target: PHQ-9 >= 5 (at least mild depressive symptoms, Kroenke et al. 2001 cut-off).
Secondary (sensitivity analysis only): PHQ-9 >= 10.

Every split, CV fold and bootstrap resample is grouped by participant (deviceId).
Reuses the raw data files and the cached minute-HR reduction; does not import the older
train_hrv_phq9_*.py scripts because their day-1 anchor (first sensor row per device) is
misaligned with the survey calendar (see Stage 1 / Stage 2 findings).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
DATA = REPO / "research_datasets/old/hrv_sleep_diaries"
SURVEY_CSV = DATA / "survey.csv"
SENSOR_CSV = DATA / "sensor_hrv.csv"
DIARY_CSV = DATA / "sleep_diary.csv"
CACHE = SCRIPTS / ".cache"
HRM_MINUTE_CACHE = CACHE / "hrv_phq9_protocol_hrm_minute.pkl"
SEGMENT_CACHE = CACHE / "hrv_phq9_protocol_segments.pkl"

STUDY_DAY1 = pd.Timestamp("2021-03-09")  # first sleep-diary date, identical for all 49 participants
STUDY_DAYS = 28
KST_OFFSET_MS = 9 * 3600 * 1000
TARGET_CUTOFF = 5
SECONDARY_CUTOFF = 10
SEED = 42
N_BOOT = 2000
OUTER_SPLITS = 5
OUTER_REPEATS = 5
INNER_SPLITS = 3
CLINICAL_TARGET_SENS = 0.80

SENSOR_CHANNELS = [
    "HR", "ibi", "sdnn", "sdsd", "rmssd", "pnn20", "pnn50", "lf", "hf", "lf/hf",
    "acc_x_avg", "acc_y_avg", "acc_z_avg", "grv_x_avg", "grv_y_avg", "grv_z_avg", "grv_w_avg",
    "gyr_x_avg", "gyr_y_avg", "gyr_z_avg", "steps", "distance", "calories", "light_avg",
]
DIARY_CHANNELS = ["sleep_duration", "in_bed_duration", "sleep_latency", "sleep_efficiency", "waso", "wakeup@night"]
BASELINE_COLS = ["PHQ9_1", "GAD7_1", "ISI_1", "MEQ"]
DEMO_COLS = ["sex", "age", "marriage", "occupation", "smartwatch", "regular", "exercise",
             "coffee", "smoking", "drinking", "height", "weight"]
# Measured at the same time as (or are) the outcome -> never features.
LEAK_COLS = ["PHQ9_2", "PHQ9_F", "GAD7_2", "GAD7_F", "ISI_2", "ISI_F"]


# ----------------------------------------------------------------------------------------
# data
def load_sensor_rows() -> pd.DataFrame:
    s = pd.read_csv(SENSOR_CSV)
    # Artifact rule (Stage 1 finding): lf / hf contain numerically impossible powers up to 1e47
    # (614 / 695 rows > 1e6 ms^2). Those values are set to NaN and lf / hf are log1p-transformed,
    # so one artifact row cannot dominate a segment mean. Applied identically to every row
    # (a fixed rule, not fit on data), so it is not a train/test leak.
    for c in ("lf", "hf"):
        s.loc[s[c] > 1e6, c] = np.nan
        s[c] = np.log1p(s[c])
    s.loc[~np.isfinite(s["lf/hf"]), "lf/hf"] = np.nan
    local = pd.to_datetime(s["ts_start"] + KST_OFFSET_MS, unit="ms")
    s["kst_date"] = local.dt.normalize()
    s["kst_hour"] = local.dt.hour
    s["study_day"] = (s["kst_date"] - STUDY_DAY1).dt.days + 1
    return s


def calendar_rows(s: pd.DataFrame) -> pd.DataFrame:
    r = s[(s["study_day"] >= 1) & (s["study_day"] <= STUDY_DAYS)].copy()
    r["segment"] = np.where(r["study_day"] <= 14, "A", "B")
    return r


def _agg_sensor(r: pd.DataFrame) -> pd.DataFrame:
    g = r.groupby(["deviceId", "segment"])
    mean = g[SENSOR_CHANNELS].mean().add_prefix("s_mean_")
    sd = g[SENSOR_CHANNELS].std().add_prefix("s_sd_")
    out = mean.join(sd)
    out["s_n_rows"] = g.size()
    out["s_n_days"] = g["study_day"].nunique()
    out["s_missingness_mean"] = g["missingness_score"].mean()
    return out


def _agg_diurnal(r: pd.DataFrame) -> pd.DataFrame:
    r = r.copy()
    r["acc_mag"] = np.sqrt(r["acc_x_avg"] ** 2 + r["acc_y_avg"] ** 2 + r["acc_z_avg"] ** 2)
    night = r[r["kst_hour"] < 6]
    day = r[(r["kst_hour"] >= 9) & (r["kst_hour"] < 21)]
    cols = ["HR", "rmssd", "acc_mag", "light_avg"]
    n = night.groupby(["deviceId", "segment"])[cols].mean().add_prefix("d_night_")
    d = day.groupby(["deviceId", "segment"])[cols].mean().add_prefix("d_day_")
    out = n.join(d, how="outer")
    out["d_hr_day_minus_night"] = out["d_day_HR"] - out["d_night_HR"]
    out["d_rmssd_night_minus_day"] = out["d_night_rmssd"] - out["d_day_rmssd"]
    # day-to-day variability of daily mean HR (circadian/behavioural regularity)
    dm = r.groupby(["deviceId", "segment", "study_day"])["HR"].mean()
    out["d_daily_hr_sd"] = dm.groupby(level=[0, 1]).std()
    return out


def _agg_diary() -> pd.DataFrame:
    d = pd.read_csv(DIARY_CSV)
    d["study_day"] = (pd.to_datetime(d["date"]) - STUDY_DAY1).dt.days + 1
    d = d[(d["study_day"] >= 1) & (d["study_day"] <= STUDY_DAYS)].copy()
    d["segment"] = np.where(d["study_day"] <= 14, "A", "B")
    d = d.rename(columns={"userId": "deviceId"})
    g = d.groupby(["deviceId", "segment"])
    return g[DIARY_CHANNELS].mean().add_prefix("sl_mean_").join(g[DIARY_CHANNELS].std().add_prefix("sl_sd_"))


def build_segments(force: bool = False) -> pd.DataFrame:
    """One row per participant-segment with label, scores and every feature group."""
    if SEGMENT_CACHE.exists() and not force:
        return pd.read_pickle(SEGMENT_CACHE)
    survey = pd.read_csv(SURVEY_CSV)
    rows = calendar_rows(load_sensor_rows())
    idx = pd.MultiIndex.from_product([sorted(survey["deviceId"]), ["A", "B"]], names=["deviceId", "segment"])
    seg = pd.DataFrame(index=idx).join(_agg_sensor(rows)).join(_agg_diurnal(rows)).join(_agg_diary()).reset_index()
    seg = seg.merge(survey, on="deviceId", how="left")
    seg["phq9_target_score"] = np.where(seg["segment"] == "A", seg["PHQ9_2"], seg["PHQ9_F"])
    seg["y"] = (seg["phq9_target_score"] >= TARGET_CUTOFF).astype(int)
    seg["y10"] = (seg["phq9_target_score"] >= SECONDARY_CUTOFF).astype(int)
    seg["s_n_rows"] = seg["s_n_rows"].fillna(0)
    seg = seg.sort_values(["deviceId", "segment"]).reset_index(drop=True)
    seg.to_pickle(SEGMENT_CACHE)
    return seg


def feature_groups(seg: pd.DataFrame) -> dict[str, list[str]]:
    sensor = [c for c in seg.columns if c.startswith("s_")]
    diurnal = [c for c in seg.columns if c.startswith("d_")]
    diary = [c for c in seg.columns if c.startswith("sl_")]
    return {"sensor": sensor, "diurnal": diurnal, "diary": diary,
            "baseline": BASELINE_COLS + DEMO_COLS, "phq9_1": ["PHQ9_1"]}


ARMS = {
    "S_sensor": ["sensor"],
    "SD_sensor_diurnal": ["sensor", "diurnal"],
    "SL_sleep_diary": ["diary"],
    "P_phq9_1_only": ["phq9_1"],
    "Q_baseline_questionnaires_demo": ["baseline"],
    "SDQ_sensor_diurnal_baseline": ["sensor", "diurnal", "baseline"],
    "ALL_everything": ["sensor", "diurnal", "diary", "baseline"],
}


def arm_columns(seg: pd.DataFrame, arm: str) -> list[str]:
    fg = feature_groups(seg)
    cols: list[str] = []
    for g in ARMS[arm]:
        cols += [c for c in fg[g] if c not in cols]
    for c in cols:
        assert c not in LEAK_COLS
    return cols


# ----------------------------------------------------------------------------------------
# models (SMALL-N: low capacity, <= 12 configurations each)
def _lr():
    return Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler()),
                     ("m", LogisticRegression(class_weight="balanced", max_iter=5000, solver="liblinear"))])


def model_specs(seed: int = SEED) -> dict:
    rng = np.random.RandomState(seed)
    specs = {
        "LR": (_lr(), [{"m__C": c, "m__penalty": p} for c in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] for p in ["l1", "l2"]][:12]),
        "RF": (Pipeline([("imp", SimpleImputer()), ("m", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=1))]),
               [{"m__max_depth": d, "m__min_samples_leaf": l} for d in [2, 4, None] for l in [1, 3, 6]]),
        "ET": (Pipeline([("imp", SimpleImputer()), ("m", ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=1))]),
               [{"m__max_depth": d, "m__min_samples_leaf": l} for d in [2, 4, None] for l in [1, 3, 6]]),
        "HGB": (Pipeline([("imp", SimpleImputer()), ("m", HistGradientBoostingClassifier(class_weight="balanced", random_state=seed))]),
                [{"m__max_depth": d, "m__learning_rate": lr, "m__max_iter": n, "m__min_samples_leaf": 5}
                 for d in [2, 3] for lr in [0.03, 0.1] for n in [50, 150]]),
        "MLP": (Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler()),
                          ("m", MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, random_state=seed))]),
                [{"m__alpha": a, "m__hidden_layer_sizes": h} for a in [0.01, 0.1, 1.0, 10.0] for h in [(8,), (16,), (32, 16)]][:12]),
    }
    try:
        import lightgbm as lgb
        space = {"num_leaves": [15, 31, 63, 127], "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                 "feature_fraction": list(np.linspace(0.6, 1.0, 5)), "bagging_fraction": list(np.linspace(0.6, 1.0, 5)),
                 "min_child_samples": [5, 10, 20, 35, 50], "n_estimators": [50, 100, 150, 200, 300],
                 "max_depth": [3, 4, 5, 6, -1], "drop_rate": [0.05, 0.1, 0.2, 0.3]}
        grid = [{"m__" + k: v for k, v in p.items()} for p in ParameterSampler(space, n_iter=10, random_state=rng)]
        specs["LGBM_DART"] = (Pipeline([("m", lgb.LGBMClassifier(boosting_type="dart", class_weight="balanced", bagging_freq=1,
                                                                  random_state=seed, verbose=-1, n_jobs=1))]), grid)
    except ImportError:
        pass
    try:
        import xgboost as xgb
        xspace = {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1, 0.3], "n_estimators": [50, 150, 300],
                  "subsample": [0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5]}
        xgrid = [{"m__" + k: v for k, v in p.items()} for p in ParameterSampler(xspace, n_iter=10, random_state=rng)]
        specs["XGB"] = (Pipeline([("m", xgb.XGBClassifier(tree_method="hist", random_state=seed, n_jobs=1, verbosity=0))]), xgrid)
    except ImportError:
        pass
    return specs


def _fit(est, X, y):
    name = est.steps[-1][0]
    kw = {}
    m = est.steps[-1][1]
    cls = type(m).__name__
    if cls in ("MLPClassifier", "XGBClassifier"):
        kw[f"{name}__sample_weight"] = compute_sample_weight("balanced", y)
    est.fit(X, y, **kw)
    return est


def _proba(est, X):
    return est.predict_proba(X)[:, 1]


def safe_auc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def inner_select(base, grid, X, y, groups, seed):
    """Pick hyperparameters by inner grouped 3-fold mean AUROC (train part only)."""
    if len(grid) == 1:
        return grid[0], np.nan
    cv = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    folds = list(cv.split(X, y, groups))
    best, best_s = grid[0], -np.inf
    for params in grid:
        scores = []
        for tr, va in folds:
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                continue
            est = clone(base).set_params(**params)
            _fit(est, X[tr], y[tr])
            scores.append(roc_auc_score(y[va], _proba(est, X[va])))
        s = np.mean(scores) if scores else -np.inf
        if s > best_s:
            best, best_s = params, s
    return best, best_s


def outer_folds(y, groups, repeat: int):
    cv = StratifiedGroupKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED + repeat)
    folds = list(cv.split(np.zeros(len(y)), y, groups))
    for tr, te in folds:
        assert set(groups[tr]).isdisjoint(set(groups[te])), "participant leak between train and test"
    return folds


def nested_cv(base, grid, X, y, groups, repeats: int = OUTER_REPEATS, fixed_params: dict | None = None,
              return_inner_oof: bool = False):
    """Repeated nested grouped CV. Returns OOF matrix (n_rows x repeats), per-repeat pooled AUROC,
    chosen params per fold, and optionally inner OOF predictions for threshold selection."""
    oof = np.full((len(y), repeats), np.nan)
    chosen, inner_info = [], []
    for r in range(repeats):
        for k, (tr, te) in enumerate(outer_folds(y, groups, r)):
            if fixed_params is not None:
                params = fixed_params
            else:
                params, _ = inner_select(base, grid, X[tr], y[tr], groups[tr], seed=SEED + 100 * r + k)
            chosen.append({"repeat": r, "fold": k, "params": {k2: (v if not isinstance(v, (np.floating, np.integer)) else v.item()) for k2, v in params.items()}})
            est = clone(base).set_params(**params)
            _fit(est, X[tr], y[tr])
            oof[te, r] = _proba(est, X[te])
            if return_inner_oof:
                icv = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + 100 * r + k)
                ip = np.full(len(tr), np.nan)
                for itr, iva in icv.split(X[tr], y[tr], groups[tr]):
                    e2 = clone(base).set_params(**params)
                    _fit(e2, X[tr][itr], y[tr][itr])
                    ip[iva] = _proba(e2, X[tr][iva])
                inner_info.append({"repeat": r, "fold": k, "train_idx": tr, "test_idx": te, "inner_oof": ip})
    per_repeat = [safe_auc(y, oof[:, r]) for r in range(repeats)]
    out = {"oof": oof, "per_repeat_auroc": per_repeat, "mean_auroc": float(np.nanmean(per_repeat)),
           "sd_auroc": float(np.nanstd(per_repeat)), "chosen": chosen}
    if return_inner_oof:
        out["inner"] = inner_info
    return out


# ----------------------------------------------------------------------------------------
# participant-level (cluster) bootstrap
def _cluster_index(groups):
    uniq = np.unique(groups)
    pos = {g: np.where(groups == g)[0] for g in uniq}
    return uniq, pos


def boot_ci(y, p, groups, metric="auroc", n_boot: int = N_BOOT, seed: int = SEED):
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    uniq, pos = _cluster_index(groups)
    rng = np.random.RandomState(seed)
    f = roc_auc_score if metric == "auroc" else average_precision_score
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ix = np.concatenate([pos[g] for g in pick])
        if len(np.unique(y[ix])) < 2:
            continue
        vals.append(f(y[ix], p[ix]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], len(vals)


def paired_boot_diff(y, pa, pb, groups, n_boot: int = N_BOOT, seed: int = SEED, metric="auroc"):
    """CI of metric(a) - metric(b) on the SAME participant resample."""
    y, pa, pb, groups = map(np.asarray, (y, pa, pb, groups))
    uniq, pos = _cluster_index(groups)
    rng = np.random.RandomState(seed)
    f = roc_auc_score if metric == "auroc" else average_precision_score
    d = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ix = np.concatenate([pos[g] for g in pick])
        if len(np.unique(y[ix])) < 2:
            continue
        d.append(f(y[ix], pa[ix]) - f(y[ix], pb[ix]))
    point = f(y, pa) - f(y, pb)
    ci = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    verdict = "better" if ci[0] > 0 else ("worse" if ci[1] < 0 else "not distinguishable")
    return {"diff": float(point), "ci": ci, "verdict": verdict, "n_valid_resamples": len(d)}


def summarize_oof(y, oof, groups):
    """Repeat-averaged OOF prediction -> point AUROC/AUPRC + participant bootstrap CIs."""
    p = np.nanmean(oof, axis=1)
    auc_ci, nb = boot_ci(y, p, groups, "auroc")
    ap_ci, _ = boot_ci(y, p, groups, "auprc")
    return {"auroc_repeat_avg_pred": float(roc_auc_score(y, p)), "auroc_ci": auc_ci,
            "auprc": float(average_precision_score(y, p)), "auprc_ci": ap_ci,
            "pos_rate_chance_auprc": float(np.mean(y)), "n_valid_resamples": nb}


# ----------------------------------------------------------------------------------------
# misc
def r4(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [r4(v) for v in x]
    if isinstance(x, dict):
        return {k: r4(v) for k, v in x.items()}
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return r4(x.tolist())
    return x


def write_json(name: str, obj) -> Path:
    p = SCRIPTS / name
    p.write_text(json.dumps(r4(obj), indent=2, default=str))
    return p


def load_json(name: str):
    return json.loads((SCRIPTS / name).read_text())


def Xy(seg: pd.DataFrame, cols: list[str], target: str = "y"):
    X = seg[cols].to_numpy(dtype=np.float64)
    X[~np.isfinite(X)] = np.nan
    return X, seg[target].to_numpy(dtype=int), seg["deviceId"].to_numpy()
