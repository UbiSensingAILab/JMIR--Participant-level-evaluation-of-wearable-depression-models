#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), derived from the DiSCover Project
# (NCT03421223), CC BY-NC 4.0 — a separate institution's protocol,
# independent of this project's own ethics protocol (REB25-1587_MOD1).
# `research_datasets/` is gitignored; the parquet is local research data
# and is never committed to this repo.
"""PSYCHE-D severity-change Phase 2 model tuning + model-family comparison.

DIAGNOSTIC ONLY. A bounded, genuine attempt to raise the real predictive
power (AUROC) of the PSYCHE-D worsened-depression Phase 2 classifier —
by nested-CV hyperparameter tuning of the LightGBM-DART model and by
comparing alternative model families on the identical feature set. No
production pipeline, dashboard, frontend or API code changes.

REUSE, NOT REIMPLEMENTATION. Following this repo's cross-script convention
(`threshold_sweep_psyche_d_severity_change.py:52-53`,
`sys.path.insert(0, str(Path(__file__).resolve().parent))` then plain module
imports), this script imports and calls — never copies — `load_raw`,
`build_phase1_input_cols`, `build_phase2_generated_features`,
`phase1_dart_factory`, `phase2_dart_factory`, `summarize`, `PROBA_COLS`,
`LEAKAGE_COLS`, `RANDOM_STATE`, `N_SPLITS` from
`audit_psyche_d_baseline_reproduction` (`REPRO`); `confusion_at_threshold`,
`CITED_REFERENCE`, `RECONCILIATION_TOLERANCE`, `TARGET_ACCURACIES` from
`threshold_sweep_psyche_d_severity_change`; and `bootstrap_ci`,
`N_BOOTSTRAP` from `train_psyche_d_severity_change`. All three source
modules stay byte-identical — reuse is import-only. There is no existing
paired two-vector bootstrap-delta helper anywhere in the repo (`bootstrap_ci`
scores one vector), so `paired_bootstrap_auroc_delta()` below is genuinely
new code, not a missed reuse.

Phase 1 is out of scope: every arm shares the identical Phase 1 path (fixed
`phase1_dart_factory()` hyperparameters, a fold-local train-participants-only
fit, inference-only intermediate-month generation) computed ONCE per outer
fold. Only Phase 2 is tuned or swapped, across seven arms fit on the
identical 163-column Phase 2 design matrix under the identical
`StratifiedGroupKFold(5)` outer protocol grouped by `user_id`:

  1. baseline_dart_fixed    — REPRO.phase2_dart_factory() unchanged, no
                               search. Regression guard + paired-comparison
                               reference.
  2. dart_nested_tuned      — LightGBM-DART, nested-CV random search.
  3. hist_gradient_boosting — sklearn HistGradientBoostingClassifier
                               (substitute for the uninstalled XGBoost; no
                               new dependency), nested-CV random search.
  4. random_forest          — RandomForestClassifier, fixed config.
  5. extra_trees            — ExtraTreesClassifier, fixed config.
  6. logistic_regression_l2 — scaled, regularized LogisticRegression,
                               nested-CV grid search over C.
  7. ensemble_diverse3      — PRE-REGISTERED (here, before any result is
                               seen) uniform mean of the pooled OOF
                               probabilities of arms 2, 3 and 6, chosen on
                               family-diversity grounds. This membership
                               MUST NOT be re-chosen after seeing results —
                               picking "the best 3 arms" by the same OOF
                               you then report is selection-on-test bias.

DISCLOSED SIMPLIFICATION: the inner hyperparameter search reuses the outer
fold's single Phase 1 model rather than refitting Phase 1 per inner split
(Phase 1 is explicitly out of scope, and refitting it per inner split would
multiply the ~7.2s prep cost by the number of inner folds for no benefit to
the question being asked). That outer-fold Phase 1 model was trained on the
outer TRAIN participants, which include the inner-validation participants,
so inner-loop model selection carries a small optimism. It cannot leak into
any reported number: the outer fold's TEST participants' labels never enter
Phase 1 training, Phase 2 training, or hyperparameter selection — enforced
by in-code assertions inside every inner split, not just implied by
construction.

VERDICT RULE (derived by code, never hardcoded): an arm counts as a real
improvement ONLY when its measured pooled-OOF AUROC exceeds the cited
reference 0.6529684832943685 AND its paired bootstrap 95% delta CI versus
the measured `baseline_dart_fixed` arm (same run, same resampled indices)
excludes zero. Headline metrics come only from the full 10866-row pooled
OOF, never a fold subset.

No participant ID, per-participant row, or per-row export is ever written to
the results JSON or the markdown report — aggregate metrics only, per this
dataset's pre-registered no-per-participant-export rule.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_psyche_d_baseline_reproduction as REPRO  # noqa: E402
from threshold_sweep_psyche_d_severity_change import (  # noqa: E402
    CITED_REFERENCE,
    RECONCILIATION_TOLERANCE,
    TARGET_ACCURACIES,
    confusion_at_threshold,
)
from train_psyche_d_severity_change import N_BOOTSTRAP, bootstrap_ci  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent

# Cited verbatim from scripts/threshold_sweep_psyche_d_severity_change_results.json
# — the baseline arm's max-achievable-accuracy regression guard.
CITED_MAX_ACCURACY_ANCHOR = {
    "majority_baseline_accuracy": 0.7927480213510031,
    "max_accuracy": 0.793208172280508,
    "max_accuracy_threshold": 0.77,
    "sensitivity_at_max_accuracy": 0.012877442273534635,
    "source": "scripts/threshold_sweep_psyche_d_severity_change_results.json",
}

PRE_REGISTERED_ENSEMBLE_MEMBERS = ["dart_nested_tuned", "hist_gradient_boosting", "logistic_regression_l2"]
FITTED_ARM_NAMES = [
    "baseline_dart_fixed",
    "dart_nested_tuned",
    "hist_gradient_boosting",
    "random_forest",
    "extra_trees",
    "logistic_regression_l2",
]
ALL_ARM_NAMES = FITTED_ARM_NAMES + ["ensemble_diverse3"]

ARM_DESCRIPTIONS = {
    "baseline_dart_fixed": "REPRO.phase2_dart_factory() unchanged, no search — regression guard and paired-comparison reference.",
    "dart_nested_tuned": "lgb.LGBMClassifier(boosting_type='dart'), nested-CV random search over num_leaves/learning_rate/feature_fraction/bagging_fraction(bagging_freq=1 fixed)/min_child_samples/n_estimators/max_depth/drop_rate.",
    "hist_gradient_boosting": "sklearn HistGradientBoostingClassifier (substitute for uninstalled XGBoost — no new dependency), nested-CV random search over learning_rate/max_leaf_nodes/min_samples_leaf/l2_regularization/max_iter.",
    "random_forest": "RandomForestClassifier(n_estimators=300, class_weight='balanced'), fixed config, no search — disproportionately expensive per-fit for a family already measured weakest.",
    "extra_trees": "ExtraTreesClassifier(n_estimators=300, class_weight='balanced'), fixed config, no search.",
    "logistic_regression_l2": "StandardScaler + LogisticRegression(class_weight='balanced'), nested-CV exhaustive search over a small log-spaced C grid.",
    "ensemble_diverse3": "PRE-REGISTERED uniform mean of pooled OOF probabilities: dart_nested_tuned + hist_gradient_boosting + logistic_regression_l2 (family-diversity grounds, chosen before any result was seen).",
}

# Documented, deterministic per-arm offset used to build each arm's inner-
# search seed: seed = RANDOM_STATE * 10_000 + ARM_SEED_OFFSET[arm] * 100 + fold_idx
# (fold_idx is 0-based) — same seed-construction convention as
# simulate_psyche_d_fl_soft_voting.py's partition_sites().
ARM_SEED_OFFSET = {"dart_nested_tuned": 1, "hist_gradient_boosting": 2, "logistic_regression_l2": 3}

DART_PARAM_DIST = {
    "num_leaves": [15, 31, 63, 127],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "feature_fraction": [0.6, 0.7, 0.8, 0.9, 1.0],
    "bagging_fraction": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_samples": [5, 10, 20, 30, 50],
    "n_estimators": [50, 100, 150, 200, 300],
    "max_depth": [3, 4, 5, 6, -1],
    "drop_rate": [0.05, 0.1, 0.15, 0.2, 0.3],
}
HGB_PARAM_DIST = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "max_leaf_nodes": [15, 31, 63, 127, None],
    "min_samples_leaf": [5, 10, 20, 30, 50],
    "l2_regularization": [0.0, 0.1, 0.5, 1.0, 2.0],
    "max_iter": [50, 100, 150, 200, 300],
}
# Small log-spaced grid — exhaustively covered since LR_INNER_DRAWS ==
# len(this grid), not a CLI flag (the 0.05s/fit makes the full grid nearly
# free, per the planner's measured per-fit runtimes).
LR_PARAM_DIST = {"logisticregression__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
LR_INNER_DRAWS = len(LR_PARAM_DIST["logisticregression__C"])


def make_dart_estimator(params: dict):
    return lgb.LGBMClassifier(
        boosting_type="dart", random_state=REPRO.RANDOM_STATE, verbosity=-1, n_jobs=-1, bagging_freq=1, **params,
    )


def make_hgb_estimator(params: dict):
    return HistGradientBoostingClassifier(random_state=REPRO.RANDOM_STATE, **params)


def make_lr_estimator(params: dict):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=REPRO.RANDOM_STATE))
    pipe.set_params(**params)
    return pipe


def inner_search(
    name: str,
    estimator_fn,
    param_dist: dict,
    n_draws: int,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    inner_splits: int,
    seed: int,
    outer_test_participants: set,
    use_sample_weight: bool,
) -> tuple[dict, list[dict]]:
    """Nested-CV inner hyperparameter search. `X_tr`/`y_tr`/`groups_tr` are
    already the OUTER-TRAIN rows only. Draws `n_draws` parameter dicts with
    `ParameterSampler(param_dist, n_iter=n_draws, random_state=seed)`, scores
    each on an inner `StratifiedGroupKFold(inner_splits)` split of the
    outer-train rows, fitting a fold-local imputer + (sample_weight or
    class_weight-balanced) model per inner split. Returns the best mean-
    inner-AUROC params plus the full per-draw record."""
    sampler = ParameterSampler(param_dist, n_iter=n_draws, random_state=seed)
    inner_skf = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=REPRO.RANDOM_STATE)
    n_tr = len(y_tr)
    records = []
    for draw_idx, params in enumerate(sampler):
        inner_aurocs = []
        n_inner_scored = 0
        for inner_tr_pos, inner_val_pos in inner_skf.split(np.zeros(n_tr), y_tr, groups=groups_tr):
            inner_train_participants = set(groups_tr[inner_tr_pos])
            inner_val_participants = set(groups_tr[inner_val_pos])
            assert not (inner_train_participants & inner_val_participants), (
                f"{name} draw {draw_idx}: inner train/val participant overlap"
            )
            assert not (inner_train_participants & outer_test_participants), (
                f"{name} draw {draw_idx}: inner-train participants intersect the outer fold's held-out test participants"
            )
            assert not (inner_val_participants & outer_test_participants), (
                f"{name} draw {draw_idx}: inner-val participants intersect the outer fold's held-out test participants"
            )

            X_inner_tr_raw = X_tr.iloc[inner_tr_pos]
            X_inner_val_raw = X_tr.iloc[inner_val_pos]
            y_inner_tr = y_tr[inner_tr_pos]
            y_inner_val = y_tr[inner_val_pos]
            if len(np.unique(y_inner_val)) < 2:
                continue

            inner_imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
            X_inner_tr = inner_imputer.fit_transform(X_inner_tr_raw)
            X_inner_val = inner_imputer.transform(X_inner_val_raw)

            est = estimator_fn(params)
            if use_sample_weight:
                sw_inner = compute_sample_weight(class_weight="balanced", y=y_inner_tr)
                est.fit(X_inner_tr, y_inner_tr, sample_weight=sw_inner)
            else:
                est.fit(X_inner_tr, y_inner_tr)
            proba_inner = est.predict_proba(X_inner_val)[:, 1]
            inner_aurocs.append(float(roc_auc_score(y_inner_val, proba_inner)))
            n_inner_scored += 1
        mean_auroc = float(np.mean(inner_aurocs)) if inner_aurocs else float("nan")
        records.append({
            "draw": draw_idx,
            "params": {k: (v if v is None else (v.item() if hasattr(v, "item") else v)) for k, v in params.items()},
            "mean_inner_auroc": mean_auroc,
            "n_inner_folds_scored": n_inner_scored,
        })

    valid_records = [r for r in records if not np.isnan(r["mean_inner_auroc"])]
    assert valid_records, f"{name}: no inner-search draw produced a valid mean inner AUROC"
    best = max(valid_records, key=lambda r: r["mean_inner_auroc"])
    return best["params"], records


def paired_bootstrap_auroc_delta(y: np.ndarray, proba_arm: np.ndarray, proba_baseline: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = REPRO.RANDOM_STATE) -> dict:
    """Per draw, resample row indices ONCE and score BOTH probability
    vectors on those same indices — skipping draws where the resample is
    single-class. New code: no paired two-vector delta helper exists
    anywhere in the repo (`bootstrap_ci` scores one vector)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y[idx]
        if len(np.unique(yt)) < 2:
            continue
        auroc_arm = roc_auc_score(yt, proba_arm[idx])
        auroc_baseline = roc_auc_score(yt, proba_baseline[idx])
        deltas.append(auroc_arm - auroc_baseline)
    if not deltas:
        return {"mean_delta": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_draws": 0}
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"mean_delta": float(np.mean(deltas)), "ci_lo": float(lo), "ci_hi": float(hi), "n_draws": len(deltas)}


def derive_verdict(arm_name: str, arm_auroc: float, paired_delta: dict) -> dict:
    """Verdict is computed from measured values only, never hardcoded. An
    arm is a real improvement ONLY when its AUROC exceeds the cited
    reference AND its paired bootstrap delta CI vs. the measured baseline
    arm excludes zero."""
    beats_cited_auroc = arm_auroc > CITED_REFERENCE["auroc"]
    ci_excludes_zero = (not np.isnan(paired_delta["ci_lo"])) and (paired_delta["ci_lo"] > 0 or paired_delta["ci_hi"] < 0)
    is_improvement = bool(beats_cited_auroc and ci_excludes_zero)
    delta_str = f"{paired_delta['mean_delta']:+.4f} (95% CI [{paired_delta['ci_lo']:+.4f}, {paired_delta['ci_hi']:+.4f}], n={paired_delta['n_draws']})"
    if is_improvement:
        verdict = "improvement"
        text = (
            f"{arm_name}: AUROC {arm_auroc:.4f} exceeds the cited baseline {CITED_REFERENCE['auroc']:.4f}, and its "
            f"paired bootstrap AUROC delta vs. the measured baseline arm is {delta_str}, which excludes zero — "
            f"REAL IMPROVEMENT."
        )
    else:
        verdict = "not distinguishable from baseline"
        reason = "AUROC does not exceed the cited baseline" if not beats_cited_auroc else "the paired bootstrap delta CI spans zero"
        text = (
            f"{arm_name}: AUROC {arm_auroc:.4f} vs. cited baseline {CITED_REFERENCE['auroc']:.4f} "
            f"(point-estimate delta {arm_auroc - CITED_REFERENCE['auroc']:+.4f}); paired bootstrap delta vs. the "
            f"measured baseline arm is {delta_str} — {reason} — NOT DISTINGUISHABLE FROM BASELINE."
        )
    return {
        "verdict": verdict,
        "text": text,
        "beats_cited_auroc_point_estimate": bool(beats_cited_auroc),
        "paired_delta_ci_excludes_zero": bool(ci_excludes_zero),
    }


def compute_threshold_sweep(y: np.ndarray, oof: np.ndarray, thresholds: np.ndarray) -> dict:
    """Threshold-vs-accuracy-vs-sensitivity sweep over an already-computed
    pooled OOF probability vector — pure post-hoc arithmetic (no
    retraining). Extracted out of the body of `compute_arm_metrics` below
    (Addendum 2 to `.planning/debug/psyche-d-fl-experiment-design.md`,
    "honest accuracy-ceiling disclosure across all FL arms") so this exact
    sweep logic can be IMPORTED and reused by the arm-B FL simulation
    scripts (`simulate_psyche_d_fl_soft_voting_arm_b.py`,
    `simulate_psyche_d_fedavg_logreg_arm_b.py`,
    `simulate_psyche_d_fed_histogram_gbdt_arm_b.py`) instead of being
    reimplemented. This is a pure refactor: the computation and output
    shape are byte-identical to the code this replaces — `compute_arm_metrics`
    below now just calls this function instead of inlining the same lines.
    This does NOT change any already-cited number from this script's own
    `tune_psyche_d_severity_change_models_results.json` (arm C) or the
    arm-B re-run's `tune_psyche_d_severity_change_models_arm_b_results.json`
    — same inputs still produce the same outputs."""
    sweep = [confusion_at_threshold(y, oof, thr) for thr in thresholds]
    best = max(sweep, key=lambda r: r["accuracy"])
    target_rows = []
    for target in TARGET_ACCURACIES:
        reaching = [r for r in sweep if r["accuracy"] >= target]
        if not reaching:
            target_rows.append({"target_accuracy": target, "reachable": False})
            continue
        b = max(reaching, key=lambda r: (r["sensitivity"] if not np.isnan(r["sensitivity"]) else -1))
        target_rows.append({
            "target_accuracy": target, "reachable": True, "threshold": b["threshold"],
            "accuracy": b["accuracy"], "sensitivity": b["sensitivity"], "specificity": b["specificity"],
        })
    return {
        "max_accuracy": best["accuracy"],
        "max_accuracy_threshold": best["threshold"],
        "sensitivity_at_max_accuracy": best["sensitivity"],
        "specificity_at_max_accuracy": best["specificity"],
        "target_accuracy_reachability": target_rows,
        "full_sweep": sweep,
    }


# Shared 0.01-resolution threshold grid (0.0 to 1.0 inclusive, 101 points) —
# module-level constant so `run_experiment` below and every importer of
# `compute_threshold_sweep` use the IDENTICAL grid without re-deriving it.
THRESHOLD_SWEEP_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 4)


def compute_arm_metrics(name: str, oof_full: np.ndarray, y_full: np.ndarray, coverage_mask: np.ndarray, full_coverage_expected: bool, n_boot: int, thresholds: np.ndarray) -> dict:
    if full_coverage_expected:
        assert coverage_mask.all(), f"{name}: expected full coverage but the OOF vector has unfilled positions"
        oof, y = oof_full, y_full
    else:
        oof, y = oof_full[coverage_mask], y_full[coverage_mask]
    assert not np.isnan(oof).any(), f"{name}: unfilled OOF positions within the evaluated mask"

    m = REPRO.summarize(name, oof, y)
    lo_a, hi_a, n_a = bootstrap_ci(y, oof, roc_auc_score, n_boot=n_boot)
    lo_p, hi_p, n_p = bootstrap_ci(y, oof, average_precision_score, n_boot=n_boot)
    m["auroc_ci"] = [lo_a, hi_a]
    m["auroc_ci_n_draws"] = n_a
    m["auprc_ci"] = [lo_p, hi_p]
    m["auprc_ci_n_draws"] = n_p
    m["n_rows_evaluated"] = int(len(y))
    m["full_coverage"] = bool(full_coverage_expected)

    m["threshold_sweep"] = compute_threshold_sweep(y, oof, thresholds)
    return m


def run_experiment(args: argparse.Namespace) -> dict:
    t0 = time.time()
    raw = REPRO.load_raw()
    phase1_input_cols = REPRO.build_phase1_input_cols(raw)
    phase2_base_cols = [c for c in raw.columns if c not in REPRO.LEAKAGE_COLS and c not in ("user_id", "user_mth", "ref_label")]
    assert len(phase2_base_cols) == 150

    labeled = raw[raw["ref_label"]].copy()
    participant_ids = labeled["user_id"].to_numpy()
    delta_cat = labeled["phq9_cat_end"] - labeled["phq9_cat_start"]
    y_full = (delta_cat > 0).astype(int).to_numpy()
    y_series = pd.Series(y_full, index=labeled.index)

    skf = StratifiedGroupKFold(n_splits=REPRO.N_SPLITS, shuffle=True, random_state=REPRO.RANDOM_STATE)
    n = len(y_full)
    all_folds = list(skf.split(np.zeros(n), y_full, groups=participant_ids))
    folds = all_folds[: args.max_folds]
    full_coverage_expected = len(folds) == REPRO.N_SPLITS
    thresholds = THRESHOLD_SWEEP_GRID

    oof = {name: np.full(n, np.nan) for name in FITTED_ARM_NAMES}
    coverage_mask = np.zeros(n, dtype=bool)
    per_fold_selected_params = {a: [] for a in ARM_SEED_OFFSET}
    per_fold_inner_search_records = {a: [] for a in ARM_SEED_OFFSET}
    per_fold_diagnostics = []

    for fold_idx, (tr_pos, te_pos) in enumerate(folds):
        fold_t0 = time.time()
        train_idx = labeled.index[tr_pos]
        test_idx = labeled.index[te_pos]
        train_participants = set(participant_ids[tr_pos])
        test_participants = set(participant_ids[te_pos])
        assert not (train_participants & test_participants), f"participant leakage fold {fold_idx}"
        coverage_mask[te_pos] = True

        # --- Shared Phase 1 path (REPRO's exact fixed hyperparameters, out
        # of scope for tuning) — computed ONCE per outer fold and shared by
        # every arm. ---
        phase1_train_mask = raw["ref_label"] & raw["user_id"].isin(train_participants)
        X1_train_df = raw.loc[phase1_train_mask, phase1_input_cols]
        y1_train = raw.loc[phase1_train_mask, "phq9_cat_end"].astype(int)
        imputer1 = SimpleImputer(strategy="mean", keep_empty_features=True)
        X1_train = imputer1.fit_transform(X1_train_df)
        sw1 = compute_sample_weight(class_weight="balanced", y=y1_train)
        clf1 = REPRO.phase1_dart_factory()
        clf1.fit(X1_train, y1_train, sample_weight=sw1)

        unlabeled_mask = ~raw["ref_label"]
        Xg = imputer1.transform(raw.loc[unlabeled_mask, phase1_input_cols])
        gen_proba = clf1.predict_proba(Xg)
        gen_pred = clf1.classes_[np.argmax(gen_proba, axis=1)]
        cat_end_generated = raw["phq9_cat_end"].copy()
        cat_end_generated.loc[unlabeled_mask] = gen_pred
        proba_df = pd.DataFrame(0.0, index=raw.index, columns=REPRO.PROBA_COLS)
        for i, c in enumerate(clf1.classes_):
            proba_df.loc[unlabeled_mask, REPRO.PROBA_COLS[int(c)]] = gen_proba[:, i]

        fold_labeled_idx = train_idx.append(test_idx)
        gen_feats = REPRO.build_phase2_generated_features(raw, fold_labeled_idx, cat_end_generated, proba_df)

        X2_base = raw.loc[fold_labeled_idx, phase2_base_cols]
        X2_full = pd.concat([X2_base, gen_feats], axis=1)
        y2 = y_series.loc[fold_labeled_idx]

        tr_mask2 = X2_full.index.isin(train_idx)
        te_mask2 = X2_full.index.isin(test_idx)

        X2_train_raw = X2_full.loc[tr_mask2]
        X2_test_raw = X2_full.loc[te_mask2]
        y2_train = y2.loc[tr_mask2].to_numpy()
        y2_test = y2.loc[te_mask2].to_numpy()
        groups_train_arr = raw.loc[X2_train_raw.index, "user_id"].to_numpy()

        # Fold-level shared imputer for every top-level (outer) arm fit —
        # same train rows for every arm on this fold, so one fit-transform is
        # equivalent to N separate fold-local imputer fits on identical data
        # (fold-local, never touches test rows), matching this project's
        # standing per-fold-imputer convention.
        imputer2_fold = SimpleImputer(strategy="mean", keep_empty_features=True)
        X2_train_imp = imputer2_fold.fit_transform(X2_train_raw)
        X2_test_imp = imputer2_fold.transform(X2_test_raw)
        # sample_weight=compute_sample_weight("balanced", ...) (DART/HGB) and
        # class_weight="balanced" (RandomForest/ExtraTrees/LogisticRegression)
        # compute IDENTICAL balanced weights internally — the difference is
        # call style only, not a difference in class-imbalance treatment.
        sw2 = compute_sample_weight(class_weight="balanced", y=y2_train)

        print(
            f"fold {fold_idx + 1}/{len(folds)} prep done: phase1_train_rows={int(phase1_train_mask.sum())} "
            f"phase2_train_rows={int(tr_mask2.sum())} phase2_test_rows={int(te_mask2.sum())} "
            f"elapsed={time.time() - fold_t0:.1f}s"
        )

        def report_arm(arm_name: str, proba_test: np.ndarray, params_desc, t_arm: float) -> None:
            oof[arm_name][te_pos] = proba_test
            fold_auroc = float(roc_auc_score(y2_test, proba_test)) if len(np.unique(y2_test)) == 2 else float("nan")
            per_fold_diagnostics.append({"fold": fold_idx + 1, "arm": arm_name, "fold_auroc": fold_auroc})
            print(
                f"fold {fold_idx + 1} arm={arm_name:24s} params={params_desc} "
                f"fold_auroc={fold_auroc:.4f} elapsed={time.time() - t_arm:.1f}s"
            )

        # --- Arm 1: baseline_dart_fixed ---
        t_arm = time.time()
        clf_base = REPRO.phase2_dart_factory()
        clf_base.fit(X2_train_imp, y2_train, sample_weight=sw2)
        report_arm("baseline_dart_fixed", clf_base.predict_proba(X2_test_imp)[:, 1], "fixed(n_estimators=150,max_depth=4,drop_rate=0.1)", t_arm)

        # --- Arm 2: dart_nested_tuned ---
        t_arm = time.time()
        seed = REPRO.RANDOM_STATE * 10_000 + ARM_SEED_OFFSET["dart_nested_tuned"] * 100 + fold_idx
        best_params, records = inner_search(
            "dart_nested_tuned", make_dart_estimator, DART_PARAM_DIST, args.dart_draws,
            X2_train_raw, y2_train, groups_train_arr, args.inner_splits, seed, test_participants, use_sample_weight=True,
        )
        per_fold_inner_search_records["dart_nested_tuned"].append({"fold": fold_idx + 1, "seed": seed, "records": records})
        per_fold_selected_params["dart_nested_tuned"].append({"fold": fold_idx + 1, "params": best_params})
        clf_dart = make_dart_estimator(best_params)
        clf_dart.fit(X2_train_imp, y2_train, sample_weight=sw2)
        report_arm("dart_nested_tuned", clf_dart.predict_proba(X2_test_imp)[:, 1], best_params, t_arm)

        # --- Arm 3: hist_gradient_boosting ---
        t_arm = time.time()
        seed = REPRO.RANDOM_STATE * 10_000 + ARM_SEED_OFFSET["hist_gradient_boosting"] * 100 + fold_idx
        best_params, records = inner_search(
            "hist_gradient_boosting", make_hgb_estimator, HGB_PARAM_DIST, args.hgb_draws,
            X2_train_raw, y2_train, groups_train_arr, args.inner_splits, seed, test_participants, use_sample_weight=True,
        )
        per_fold_inner_search_records["hist_gradient_boosting"].append({"fold": fold_idx + 1, "seed": seed, "records": records})
        per_fold_selected_params["hist_gradient_boosting"].append({"fold": fold_idx + 1, "params": best_params})
        clf_hgb = make_hgb_estimator(best_params)
        clf_hgb.fit(X2_train_imp, y2_train, sample_weight=sw2)
        report_arm("hist_gradient_boosting", clf_hgb.predict_proba(X2_test_imp)[:, 1], best_params, t_arm)

        # --- Arm 4: random_forest (fixed, no search — 4.30s/fit measured) ---
        t_arm = time.time()
        clf_rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=REPRO.RANDOM_STATE, n_jobs=-1)
        clf_rf.fit(X2_train_imp, y2_train)
        report_arm("random_forest", clf_rf.predict_proba(X2_test_imp)[:, 1], "fixed(n_estimators=300,class_weight=balanced)", t_arm)

        # --- Arm 5: extra_trees (fixed, no search) ---
        t_arm = time.time()
        clf_et = ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=REPRO.RANDOM_STATE, n_jobs=-1)
        clf_et.fit(X2_train_imp, y2_train)
        report_arm("extra_trees", clf_et.predict_proba(X2_test_imp)[:, 1], "fixed(n_estimators=300,class_weight=balanced)", t_arm)

        # --- Arm 6: logistic_regression_l2 ---
        t_arm = time.time()
        seed = REPRO.RANDOM_STATE * 10_000 + ARM_SEED_OFFSET["logistic_regression_l2"] * 100 + fold_idx
        best_params, records = inner_search(
            "logistic_regression_l2", make_lr_estimator, LR_PARAM_DIST, LR_INNER_DRAWS,
            X2_train_raw, y2_train, groups_train_arr, args.inner_splits, seed, test_participants, use_sample_weight=False,
        )
        per_fold_inner_search_records["logistic_regression_l2"].append({"fold": fold_idx + 1, "seed": seed, "records": records})
        per_fold_selected_params["logistic_regression_l2"].append({"fold": fold_idx + 1, "params": best_params})
        clf_lr = make_lr_estimator(best_params)
        clf_lr.fit(X2_train_imp, y2_train)
        report_arm("logistic_regression_l2", clf_lr.predict_proba(X2_test_imp)[:, 1], best_params, t_arm)

    assert coverage_mask.sum() == sum(len(te_pos) for _, te_pos in folds)

    # --- Arm 7: ensemble_diverse3 — pre-registered post-hoc uniform mean of
    # the three named arms' pooled OOF probability vectors. No refitting. ---
    oof["ensemble_diverse3"] = np.mean([oof[a] for a in PRE_REGISTERED_ENSEMBLE_MEMBERS], axis=0)

    arms_metrics = {
        name: compute_arm_metrics(name, oof[name], y_full, coverage_mask, full_coverage_expected, args.n_boot, thresholds)
        for name in ALL_ARM_NAMES
    }

    baseline_oof_eval = oof["baseline_dart_fixed"][coverage_mask] if not full_coverage_expected else oof["baseline_dart_fixed"]
    y_eval = y_full[coverage_mask] if not full_coverage_expected else y_full

    paired_deltas = {}
    verdicts = {}
    for name in ALL_ARM_NAMES:
        if name == "baseline_dart_fixed":
            continue
        arm_oof_eval = oof[name][coverage_mask] if not full_coverage_expected else oof[name]
        paired_deltas[name] = paired_bootstrap_auroc_delta(y_eval, arm_oof_eval, baseline_oof_eval, n_boot=args.n_boot, seed=REPRO.RANDOM_STATE)
        verdicts[name] = derive_verdict(name, arms_metrics[name]["auroc"], paired_deltas[name])

    baseline_auroc_delta = arms_metrics["baseline_dart_fixed"]["auroc"] - CITED_REFERENCE["auroc"]
    baseline_max_acc_delta = arms_metrics["baseline_dart_fixed"]["threshold_sweep"]["max_accuracy"] - CITED_MAX_ACCURACY_ANCHOR["max_accuracy"]
    baseline_reconciliation = {
        "measured_auroc": arms_metrics["baseline_dart_fixed"]["auroc"],
        "cited_auroc": CITED_REFERENCE["auroc"],
        "delta_auroc": baseline_auroc_delta,
        "auroc_within_tolerance": bool(abs(baseline_auroc_delta) <= RECONCILIATION_TOLERANCE) if full_coverage_expected else None,
        "measured_max_accuracy": arms_metrics["baseline_dart_fixed"]["threshold_sweep"]["max_accuracy"],
        "cited_max_accuracy": CITED_MAX_ACCURACY_ANCHOR["max_accuracy"],
        "delta_max_accuracy": baseline_max_acc_delta,
        "max_accuracy_within_tolerance": bool(abs(baseline_max_acc_delta) <= RECONCILIATION_TOLERANCE) if full_coverage_expected else None,
        "tolerance": RECONCILIATION_TOLERANCE,
        "note": "Reconciliation is only meaningful with full 5-fold coverage; smoke/reduced-fold runs will not reconcile." if not full_coverage_expected else "",
    }

    improving_arms = [name for name, v in verdicts.items() if v["verdict"] == "improvement"]
    best_arm_name = max(ALL_ARM_NAMES, key=lambda n: arms_metrics[n]["auroc"])
    best_arm = arms_metrics[best_arm_name]
    best_ts = best_arm["threshold_sweep"]
    headline = {
        "any_arm_beats_baseline": bool(improving_arms),
        "improving_arms": improving_arms,
        "best_arm_by_auroc": best_arm_name,
        "best_arm_auroc": best_arm["auroc"],
        "best_arm_max_accuracy": best_ts["max_accuracy"],
        "best_arm_max_accuracy_threshold": best_ts["max_accuracy_threshold"],
        "best_arm_sensitivity_at_max_accuracy": best_ts["sensitivity_at_max_accuracy"],
        "85pct_accuracy_reachable_for_best_arm": bool(best_ts["max_accuracy"] >= 0.85),
    }

    return {
        "library_versions": {
            "pandas": pd.__version__, "numpy": np.__version__, "lightgbm": lgb.__version__, "scikit-learn": sklearn.__version__,
        },
        "cli_budgets": {
            "dart_draws": args.dart_draws, "hgb_draws": args.hgb_draws, "lr_draws": LR_INNER_DRAWS,
            "inner_splits": args.inner_splits, "max_folds": args.max_folds, "n_boot": args.n_boot,
        },
        "n_splits": REPRO.N_SPLITS,
        "n_folds_run": len(folds),
        "full_coverage": full_coverage_expected,
        "random_state": REPRO.RANDOM_STATE,
        "arm_seed_offset": ARM_SEED_OFFSET,
        "seed_rule": "seed = RANDOM_STATE * 10_000 + ARM_SEED_OFFSET[arm] * 100 + fold_idx (fold_idx is 0-based)",
        "cited_reference": CITED_REFERENCE,
        "cited_max_accuracy_anchor": CITED_MAX_ACCURACY_ANCHOR,
        "arm_registry": {
            name: {
                "description": ARM_DESCRIPTIONS[name],
                "search_space": (
                    DART_PARAM_DIST if name == "dart_nested_tuned" else
                    HGB_PARAM_DIST if name == "hist_gradient_boosting" else
                    LR_PARAM_DIST if name == "logistic_regression_l2" else None
                ),
            }
            for name in ALL_ARM_NAMES
        },
        "pre_registered_ensemble_members": PRE_REGISTERED_ENSEMBLE_MEMBERS,
        "disclosed_simplification": (
            "The inner hyperparameter search reuses the outer fold's single Phase 1 model rather than "
            "refitting Phase 1 per inner split. That outer-fold Phase 1 model was trained on the outer TRAIN "
            "participants, which include the inner-validation participants, so inner-loop model selection "
            "carries a small optimism. It cannot leak into any reported number: the outer fold's TEST "
            "participants' labels never enter Phase 1 training, Phase 2 training, or hyperparameter selection "
            "(enforced by in-code assertions inside every inner split)."
        ),
        "per_fold_selected_params": per_fold_selected_params,
        "per_fold_inner_search_records": per_fold_inner_search_records,
        "per_fold_diagnostics_not_headline": per_fold_diagnostics,
        "arms": arms_metrics,
        "paired_delta_vs_baseline": paired_deltas,
        "verdicts": verdicts,
        "baseline_reconciliation": baseline_reconciliation,
        "headline": headline,
        "runtime_seconds": time.time() - t0,
    }


def build_report(results: dict) -> str:
    lv = results["library_versions"]
    cb = results["cli_budgets"]
    lines = []
    lines.append("# PSYCHE-D Phase 2 Model Tuning + Family Comparison — Results Report")
    lines.append("")
    lines.append(
        "PSYCHE-D dataset (Zenodo 5085146), derived from the DiSCover Project (NCT03421223), "
        "CC BY-NC 4.0 — a separate institution's protocol, independent of this project's own ethics "
        "protocol (REB25-1587_MOD1). `research_datasets/` is gitignored; the parquet is local research "
        "data and is never committed to this repo."
    )
    lines.append("")
    lines.append(
        "**Scope.** This is a diagnostic modeling experiment — a genuine, bounded attempt to raise the "
        "Phase 2 classifier's AUROC via nested-CV hyperparameter tuning and alternative model families. "
        "No production pipeline, dashboard, frontend or API code changed."
    )
    lines.append("")
    lines.append(
        f"**Library versions:** pandas={lv['pandas']} numpy={lv['numpy']} lightgbm={lv['lightgbm']} "
        f"scikit-learn={lv['scikit-learn']}"
    )
    lines.append("")

    lines.append("## Setup")
    lines.append("")
    lines.append("| Aspect | Value |")
    lines.append("|---|---|")
    lines.append(f"| Outer protocol | `StratifiedGroupKFold(n_splits={results['n_splits']}, shuffle=True, random_state={results['random_state']})`, grouped by `user_id`, identical to `audit_psyche_d_baseline_reproduction.py` |")
    lines.append(f"| Folds run | {results['n_folds_run']} of {results['n_splits']} (full coverage: {results['full_coverage']}) |")
    lines.append(f"| Phase 1 | Unchanged — fixed `phase1_dart_factory()` hyperparameters, fold-local train-participants-only fit, inference-only intermediate-month generation, computed once per outer fold and shared across all seven arms |")
    lines.append(f"| Inner search budgets | DART draws={cb['dart_draws']}, HGB draws={cb['hgb_draws']}, LR draws={cb['lr_draws']} (exhaustive small grid), inner splits={cb['inner_splits']} |")
    lines.append(f"| Inner-search seed rule | `{results['seed_rule']}` |")
    lines.append(f"| Bootstrap CIs | n_boot={cb['n_boot']}, `bootstrap_ci` (single-vector) imported from `train_psyche_d_severity_change.py`; `paired_bootstrap_auroc_delta` (new, two-vector) defined in this script |")
    lines.append(f"| Disclosed simplification | {results['disclosed_simplification']} |")
    lines.append(f"| Pre-registered ensemble | `ensemble_diverse3` = uniform mean of pooled OOF probabilities of {', '.join(results['pre_registered_ensemble_members'])} — membership fixed before any result was seen |")
    lines.append("| Substitution | `HistGradientBoostingClassifier` (scikit-learn, already installed) stands in for the uninstalled XGBoost — no new dependency added |")
    lines.append("")

    lines.append("## Arm registry")
    lines.append("")
    lines.append("| Arm | Description |")
    lines.append("|---|---|")
    for name in ALL_ARM_NAMES:
        lines.append(f"| `{name}` | {results['arm_registry'][name]['description']} |")
    lines.append("")

    lines.append("## Main comparison")
    lines.append("")
    lines.append("| Arm | AUROC | AUROC 95% CI | AUPRC | AUPRC 95% CI | Accuracy@0.5 | Sens@0.5 | Spec@0.5 | Max accuracy (threshold) | Sens. at max-acc | Δ AUROC vs. baseline (paired 95% CI) | Verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in ALL_ARM_NAMES:
        m = results["arms"][name]
        ts = m["threshold_sweep"]
        if name == "baseline_dart_fixed":
            delta_cell = "— (reference)"
            verdict_cell = "reference arm"
        else:
            pd_ = results["paired_delta_vs_baseline"][name]
            delta_cell = f"{pd_['mean_delta']:+.4f} ([{pd_['ci_lo']:+.4f}, {pd_['ci_hi']:+.4f}])"
            verdict_cell = results["verdicts"][name]["verdict"]
        lines.append(
            f"| `{name}` | {m['auroc']:.4f} | [{m['auroc_ci'][0]:.4f}, {m['auroc_ci'][1]:.4f}] | {m['auprc']:.4f} | "
            f"[{m['auprc_ci'][0]:.4f}, {m['auprc_ci'][1]:.4f}] | {m['accuracy']:.4f} | {m['sensitivity']:.4f} | "
            f"{m['specificity']:.4f} | {ts['max_accuracy']:.4f} ({ts['max_accuracy_threshold']:.2f}) | "
            f"{ts['sensitivity_at_max_accuracy']:.4f} | {delta_cell} | {verdict_cell} |"
        )
    lines.append("")

    br = results["baseline_reconciliation"]
    lines.append(
        f"**Baseline reconciliation:** measured `baseline_dart_fixed` AUROC {br['measured_auroc']:.4f} vs. cited "
        f"{br['cited_auroc']:.4f} (delta {br['delta_auroc']:+.4f}, tolerance ±{br['tolerance']}"
        f"{', within tolerance: ' + str(br['auroc_within_tolerance']) if br['auroc_within_tolerance'] is not None else ' — not evaluated on a reduced-fold run'}"
        f"); measured max-achievable accuracy {br['measured_max_accuracy']:.4f} vs. cited "
        f"{br['cited_max_accuracy']:.4f} (delta {br['delta_max_accuracy']:+.4f}"
        f"{', within tolerance: ' + str(br['max_accuracy_within_tolerance']) if br['max_accuracy_within_tolerance'] is not None else ''})."
    )
    lines.append("")

    lines.append("## Honest verdict")
    lines.append("")
    h = results["headline"]
    if h["any_arm_beats_baseline"]:
        lines.append(
            f"**Headline: {', '.join(h['improving_arms'])} beat the cited baseline AUROC "
            f"({CITED_REFERENCE['auroc']:.4f}) outside noise** — both the AUROC point-estimate condition and the "
            f"paired bootstrap delta-excludes-zero condition held. See the per-arm verdict text below for exact "
            f"magnitudes."
        )
    else:
        lines.append(
            f"**Headline: nothing beat the cited baseline AUROC ({CITED_REFERENCE['auroc']:.4f}) outside noise.** "
            f"Nested-CV tuning of LightGBM-DART and every alternative model family measured here landed at or "
            f"within noise of the baseline on this feature set. This is the correct, expected, acceptable result "
            f"of a genuine bounded attempt — it closes a real open question rather than leaving it unanswered."
        )
    lines.append("")
    for name in ALL_ARM_NAMES:
        if name == "baseline_dart_fixed":
            continue
        lines.append(f"- {results['verdicts'][name]['text']}")
    lines.append("")
    if h["85pct_accuracy_reachable_for_best_arm"]:
        lines.append(
            f"**85% accuracy question:** reachable for the best arm (`{h['best_arm_by_auroc']}`, AUROC "
            f"{h['best_arm_auroc']:.4f}) — max-achievable accuracy {h['best_arm_max_accuracy']:.4f} at threshold "
            f"{h['best_arm_max_accuracy_threshold']:.2f}, sensitivity {h['best_arm_sensitivity_at_max_accuracy']:.4f} "
            f"at that threshold."
        )
    else:
        lines.append(
            f"**85% accuracy question:** NOT reachable at any swept threshold for the best arm "
            f"(`{h['best_arm_by_auroc']}`, AUROC {h['best_arm_auroc']:.4f}) — its own max-achievable accuracy over "
            f"the 0.01 grid is {h['best_arm_max_accuracy']:.4f} at threshold {h['best_arm_max_accuracy_threshold']:.2f} "
            f"(sensitivity {h['best_arm_sensitivity_at_max_accuracy']:.4f} there)."
        )
    lines.append("")

    lines.append("## How to reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib python3 scripts/tune_psyche_d_severity_change_models.py")
    lines.append("```")
    lines.append("")
    lines.append(f"Runtime of this run: {results.get('runtime_seconds', float('nan')):.1f}s.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dart-draws", type=int, default=40)
    parser.add_argument("--hgb-draws", type=int, default=12)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--max-folds", type=int, default=REPRO.N_SPLITS)
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--out-suffix", type=str, default="")
    args = parser.parse_args()

    print(f"pandas={pd.__version__} numpy={np.__version__} lightgbm={lgb.__version__} scikit-learn={sklearn.__version__}")
    results = run_experiment(args)

    json_path = SCRIPT_DIR / f"tune_psyche_d_severity_change_models_results{args.out_suffix}.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {json_path}")

    report_path = SCRIPT_DIR / f"tune_psyche_d_severity_change_models_report{args.out_suffix}.md"
    report_path.write_text(build_report(results))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
