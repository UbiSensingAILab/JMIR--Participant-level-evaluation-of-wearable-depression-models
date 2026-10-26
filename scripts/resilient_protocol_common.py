#!/usr/bin/env python3
# RESILIENT dataset (Zenodo 16755408): a different institution's REC/IRAS-approved
# protocol (IRAS 321104), independent of this project's REB25-1587_MOD1. Treat all
# participant data as research data; never print individual rows into reports.
"""Shared helpers for the RESILIENT dataset-study-protocol run (prefix resilient_protocol).

Runs in .venv-psyche-d-fl-mlp (sklearn, lightgbm, xgboost, torch). Contains NO data
loading of raw files (that lives in stage00_01, which runs in .venv311 and reuses the
existing loaders from train_resilient_multimodal_binary_phq9.py). Everything here
operates on the cached participant-level feature table.

SMALL-N branch (protocol Stage 0): repeated nested stratified CV, outer 5-fold x 5
repeats, inner 3-fold, <= 10 hyperparameter configurations per model. One row per
participant, so StratifiedGroupKFold == StratifiedKFold; disjointness is still asserted.
"""
from __future__ import annotations

import os

# torch and LightGBM/XGBoost ship separate libomp runtimes; with >1 OpenMP thread the
# MLP step deadlocked (observed: process stuck in __kmp_join_barrier). Single-threaded
# OpenMP everywhere avoids it; the data are tiny so this costs little.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
FEATURES_PKL = SCRIPT_DIR / "resilient_protocol_features.pkl"
MOMENT_PKL = SCRIPT_DIR / "resilient_protocol_moment_embeds.pkl"

SEED = 42
N_OUTER = 5
N_REPEATS = 5
N_INNER = 3
N_CONFIGS = 10
N_BOOT = 2000
THRESH_GRID = np.linspace(0.005, 0.995, 199)
CLINICAL_TARGET_SENS = 0.80  # fallback reference sensitivity (no published model on this data)


def load_features() -> dict:
    return pickle.loads(FEATURES_PKL.read_bytes())


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    print(f"[write] {path.relative_to(REPO)}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def r4(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [r4(v) for v in x]
    if isinstance(x, float) and np.isnan(x):
        return None
    return round(float(x), 4)


# ---------------------------------------------------------------------------
# Models and search spaces (low capacity, SMALL-N)
# ---------------------------------------------------------------------------

def _scaled(est):
    return Pipeline([("imp", SimpleImputer(strategy="mean", keep_empty_features=True)),
                     ("sc", StandardScaler()), ("clf", est)])


def _imputed(est):
    return Pipeline([("imp", SimpleImputer(strategy="mean", keep_empty_features=True)), ("clf", est)])


class TorchMLP:
    """sklearn-like wrapper around the PSYCHE-D SmallMLP (in -> 64 -> 32 -> 1), reused
    from psyche_d_fedavg_vs_fedtree_mlp_model.py. Mean impute + z-score fit on train."""

    def __init__(self, epochs=80, lr=1e-3, seed=0):
        self.epochs, self.lr, self.seed = epochs, lr, seed

    def get_params(self, deep=False):
        return {"epochs": self.epochs, "lr": self.lr, "seed": self.seed}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        import torch
        import psyche_d_fedavg_vs_fedtree_mlp_model as M
        torch.set_num_threads(1)
        self.imp = SimpleImputer(strategy="mean", keep_empty_features=True).fit(X)
        self.sc = StandardScaler().fit(self.imp.transform(X))
        Z = self.sc.transform(self.imp.transform(X))
        torch.manual_seed(self.seed)
        self.model = M.SmallMLP(input_dim=Z.shape[1])
        M.train_local(self.model, Z, np.asarray(y), epochs=self.epochs, lr=self.lr, batch_size=64, seed=self.seed)
        return self

    def predict_proba(self, X):
        import psyche_d_fedavg_vs_fedtree_mlp_model as M
        p = M.predict_proba(self.model, self.sc.transform(self.imp.transform(X)))
        return np.column_stack([1 - p, p])


def model_factory(name: str, y_train: np.ndarray, seed: int = SEED):
    """Returns (base_estimator, param_space). param keys use sklearn pipeline syntax."""
    spw = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
    if name == "LogReg":
        from sklearn.linear_model import LogisticRegression
        est = _scaled(LogisticRegression(class_weight="balanced", max_iter=5000, solver="liblinear"))
        space = {"clf__C": [1e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 10, 100], "clf__penalty": ["l2", "l1"]}
    elif name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        est = _imputed(RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=seed, n_jobs=2))
        space = {"clf__max_depth": [2, 3, 5, None], "clf__min_samples_leaf": [1, 3, 5], "clf__max_features": ["sqrt", 0.3]}
    elif name == "ExtraTrees":
        from sklearn.ensemble import ExtraTreesClassifier
        est = _imputed(ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=seed, n_jobs=2))
        space = {"clf__max_depth": [2, 3, 5, None], "clf__min_samples_leaf": [1, 3, 5], "clf__max_features": ["sqrt", 0.3]}
    elif name == "HistGB":
        from sklearn.ensemble import HistGradientBoostingClassifier
        est = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
        space = {"max_depth": [2, 3], "learning_rate": [0.03, 0.1], "max_iter": [50, 150], "min_samples_leaf": [3, 5, 10]}
    elif name == "XGBoost":
        from xgboost import XGBClassifier
        est = XGBClassifier(subsample=0.8, colsample_bytree=0.8, tree_method="hist", scale_pos_weight=spw,
                            random_state=seed, n_jobs=2, eval_metric="auc", verbosity=0)
        space = {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1], "n_estimators": [50, 150], "min_child_weight": [1, 3]}
    elif name == "LightGBM-DART":
        from lightgbm import LGBMClassifier
        est = LGBMClassifier(boosting_type="dart", is_unbalance=True, random_state=seed, n_jobs=2, verbose=-1)
        # min_child_samples range lowered from the protocol's 5-50 default: ~58 training rows.
        space = {"num_leaves": [4, 8, 15], "learning_rate": [0.03, 0.05, 0.1], "n_estimators": [50, 150, 300],
                 "min_child_samples": [3, 5, 10], "drop_rate": [0.05, 0.1, 0.3], "max_depth": [2, 3, 4, -1],
                 "colsample_bytree": [0.6, 0.8, 1.0], "subsample": [0.6, 0.8, 1.0], "subsample_freq": [1]}
    elif name == "MLP":
        est = TorchMLP(seed=seed)
        space = {"epochs": [40, 80, 150], "lr": [1e-3, 3e-3]}
    else:
        raise ValueError(name)
    return est, space


MODEL_NAMES = ["LogReg", "RandomForest", "ExtraTrees", "HistGB", "XGBoost", "LightGBM-DART", "MLP"]


def _configs(space: dict, seed: int) -> list[dict]:
    n_total = int(np.prod([len(v) for v in space.values()]))
    return list(ParameterSampler(space, n_iter=min(N_CONFIGS, n_total), random_state=seed))


def _fit_predict(name, params, Xtr, ytr, Xte, seed):
    est, _ = model_factory(name, ytr, seed)
    est = clone(est) if not isinstance(est, TorchMLP) else est
    est.set_params(**params)
    est.fit(Xtr, ytr)
    return est.predict_proba(Xte)[:, 1], est


def nested_select(name, X, y, seed):
    """Inner StratifiedKFold(3) AUROC mean over <= N_CONFIGS sampled configurations.
    Returns best params and the inner out-of-fold scores of the best config (used for
    validation-only threshold selection in Stage 5)."""
    _, space = model_factory(name, y, seed)
    configs = _configs(space, seed)
    inner = StratifiedKFold(N_INNER, shuffle=True, random_state=seed)
    best, best_auc, best_oof = None, -1.0, None
    for cfg in configs:
        oof = np.zeros(len(y))
        aucs = []
        for tr, va in inner.split(X, y):
            p, _ = _fit_predict(name, cfg, X[tr], y[tr], X[va], seed)
            oof[va] = p
            aucs.append(roc_auc_score(y[va], p) if len(np.unique(y[va])) == 2 else np.nan)
        m = float(np.nanmean(aucs))
        if m > best_auc:
            best, best_auc, best_oof = cfg, m, oof
    return best, best_auc, best_oof


def repeated_nested_cv(name, X, y, uids, seed=SEED, n_repeats=N_REPEATS, with_thresholds=False, fixed_params=None):
    """Outer RepeatedStratifiedKFold (one row = one participant). Returns per-repeat OOF
    scores [n_repeats, n], per-fold records, and (optionally) validation-selected
    thresholds per outer fold from the inner OOF scores."""
    n = len(y)
    oof = np.full((n_repeats, n), np.nan)
    folds = []
    for r in range(n_repeats):
        outer = StratifiedKFold(N_OUTER, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(outer.split(X, y)):
            assert set(uids[tr]).isdisjoint(set(uids[te])), "participant leak across outer split"
            fseed = seed + 100 * r + f
            if fixed_params is None:
                params, inner_auc, inner_oof = nested_select(name, X[tr], y[tr], fseed)
            else:
                params, inner_auc, inner_oof = fixed_params, None, None
            p, _ = _fit_predict(name, params, X[tr], y[tr], X[te], fseed)
            oof[r, te] = p
            rec = {"repeat": r, "fold": f, "n_train": len(tr), "n_test": len(te),
                   "pos_train": int(y[tr].sum()), "pos_test": int(y[te].sum()),
                   "params": params, "inner_auc": inner_auc,
                   "test_auc": float(roc_auc_score(y[te], p)) if len(np.unique(y[te])) == 2 else None,
                   "score_mean_pos": float(p[y[te] == 1].mean()) if y[te].sum() else None,
                   "score_mean_neg": float(p[y[te] == 0].mean())}
            if with_thresholds and inner_oof is not None:
                rec["thresholds"] = select_thresholds(y[tr], inner_oof)
                rec["test_idx"] = te.tolist()
                rec["test_scores"] = p.tolist()
            folds.append(rec)
    assert not np.isnan(oof).any()
    return {"oof": oof, "folds": folds}


# ---------------------------------------------------------------------------
# Metrics, bootstrap, thresholds
# ---------------------------------------------------------------------------

def summarize(y, oof, n_boot=N_BOOT, seed=SEED) -> dict:
    """Headline = AUROC of the repeat-averaged OOF score (one score per participant),
    participant-level bootstrap CI. Also mean and population SD of per-repeat AUROCs."""
    s = oof.mean(axis=0)
    per_rep = [roc_auc_score(y, oof[r]) for r in range(oof.shape[0])]
    per_rep_ap = [average_precision_score(y, oof[r]) for r in range(oof.shape[0])]
    auc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)
    lo, hi, lo2, hi2 = boot_ci(y, s, n_boot, seed)
    return {"auroc": auc, "auroc_ci": [lo, hi], "auprc": ap, "auprc_ci": [lo2, hi2],
            "auroc_per_repeat_mean": float(np.mean(per_rep)), "auroc_per_repeat_sd": float(np.std(per_rep)),
            "auprc_per_repeat_mean": float(np.mean(per_rep_ap)), "chance_auprc": float(y.mean())}


def boot_ci(y, s, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    a, b = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        a.append(roc_auc_score(y[idx], s[idx]))
        b.append(average_precision_score(y[idx], s[idx]))
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)),
            float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def paired_boot_diff(y, s_a, s_b, n_boot=N_BOOT, seed=SEED) -> dict:
    """AUROC(a) - AUROC(b), same participant resample for both."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(roc_auc_score(y[idx], s_a[idx]) - roc_auc_score(y[idx], s_b[idx]))
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    delta = float(roc_auc_score(y, s_a) - roc_auc_score(y, s_b))
    verdict = "better" if lo > 0 else ("worse" if hi < 0 else "not distinguishable")
    return {"delta": delta, "ci": [lo, hi], "verdict": verdict}


def confusion_at(y, s, t) -> dict:
    pred = (s >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return {"threshold": float(t), "accuracy": (tp + tn) / len(y), "sensitivity": sens, "specificity": spec,
            "balanced_accuracy": (sens + spec) / 2, "ppv": tp / (tp + fp) if tp + fp else None,
            "npv": tn / (tn + fn) if tn + fn else None, "tp": tp, "fn": fn, "tn": tn, "fp": fp}


def select_thresholds(y, s, ref_sens=CLINICAL_TARGET_SENS) -> dict:
    """Validation-only threshold selection (inner OOF scores of the training fold)."""
    rows = [confusion_at(y, s, t) for t in THRESH_GRID]
    default = max(rows, key=lambda r: (r["balanced_accuracy"], -abs(r["threshold"] - 0.5)))["threshold"]
    maxacc = max(rows, key=lambda r: (r["accuracy"], r["sensitivity"]))["threshold"]
    grid = np.arange(0.0, 1.0 + 1e-9, 0.0025)
    ok = [t for t in grid if confusion_at(y, s, t)["sensitivity"] >= ref_sens]
    matched = float(max(ok)) if ok else 0.0
    return {"default": float(default), "max_accuracy": float(maxacc), "matched_sensitivity": float(matched)}
