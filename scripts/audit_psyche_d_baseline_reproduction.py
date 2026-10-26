#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), derived from the DiSCover Project
# (NCT03421223), CC BY-NC 4.0 — a separate institution's protocol,
# independent of this project's own ethics protocol (REB25-1587_MOD1).
# `research_datasets/` is gitignored; the parquet is local research data
# and is never committed to this repo.
"""PSYCHE-D baseline audit — PART B3/B4 (leakage-safe two-phase reproduction)
+ a permutation sanity check on the reproduction pipeline itself.

Reproduces, as closely as is reasonable for an audit (not a re-implementation
project), the original PSYCHE-D two-phase pipeline described in Makhmutova
et al. 2022 (JMIR mHealth uHealth) and implemented at
https://github.com/evidation-opensource/PSYCHE-D (`combined_pipeline.py`,
`utils.py`, inspected 2026-09-23, commit at time of inspection = `main`
HEAD). Adapted to this project's StratifiedGroupKFold(5) pooled-OOF
evaluation convention instead of the paper's single random 800-participant
holdout repeated over 5 seeds.

Simplifications vs. the original repo, disclosed explicitly rather than
hidden (Strict Rule: do not claim reproduction if required inputs are
missing, and do not silently substitute a different design):

1. Phase 1 feature list (`PHASE_1_INPUT_COLS`) is copied VERBATIM from the
   original repo's `combined_pipeline.py`. All 80 columns it references
   exist in our parquet (see `audit_psyche_d_baseline_report.md` Part B2)
   - zero missing.
2. Phase 1 / Phase 2 hyperparameters are fixed, reasonable LightGBM-DART
   settings (matching the magnitude of the repo's own commented-out
   "selected parameters" fallback path) rather than the original's
   50-iteration `RandomizedSearchCV` + `RFE` (Phase 1) / greedy
   `SequentialFeatureSelector` (Phase 2) search. Reproducing the SEARCH
   PROCEDURE itself is out of scope for an audit ("do not optimize beyond
   what's necessary for reproduction" - explicit non-goal is hyperparameter
   optimization aimed at improving the number).
3. Phase 2 feature set is this project's full 150-column wearable +
   screener + LMC set (same feature window as every other arm in this
   audit) PLUS the generated `cat_m0`/`cat_m1`/`cat_m2` +
   `proba_cat_{0-4}_m1`/`_m2` columns (163 total), rather than the
   original's `SequentialFeatureSelector`-selected 13-column subset per
   random seed. Selecting down to 13 features via a greedy CV search is a
   modeling-improvement step, not a reproduction-fidelity step, and stays
   out of scope for the same reason as (2).
4. Evaluation is pooled out-of-fold `StratifiedGroupKFold(5)` (this
   project's standing convention, applied identically to every other arm
   in this audit) instead of the paper's single random 800-participant
   train/test split repeated over 5 seeds. This changes the evaluation
   protocol, so raw numbers are NOT directly, numerically comparable to
   the published Phase-2 sensitivity/specificity - reported as
   QUALITATIVELY CONSISTENT, never claimed as a numeric match.

Leakage safety (Strict Rule - "never generate Phase 1 predictions using a
model trained on the same participant whose Phase 2 sample is being
evaluated"), enforced per fold:

- Phase 1 is trained ONLY on quarter-end (labeled) rows belonging to that
  fold's TRAIN participants.
- Phase 1 predictions for intermediate (unlabeled) months are then
  generated for ALL rows in the full 35,694-row frame (train-fold AND
  test-fold participants' intermediate months) using that fold-specific
  Phase-1 model. This mirrors the original repo's own design (Phase 1 fit
  once on `data_train`, then applied via inference to `data_train`'s AND
  `data_test`'s unlabeled rows) and is leakage-safe because inference alone
  never uses a test participant's true PHQ-9 label.
- Phase 2 is trained ONLY on that fold's TRAIN participants' labeled rows
  and evaluated ONLY on that fold's TEST participants' labeled rows.
- `assert not (train_participants & test_participants)` is checked inside
  every fold, not decorative.

A trailing permutation sanity check (`--permutation-check`) re-runs the
identical pipeline with the binary `worsened` label shuffled (Phase 1's
5-class `phq9_cat_end` target is intentionally left real, since Phase 1
never sees the binary label at all - only Phase 2's fit/eval target is
shuffled) to confirm the reproduction pipeline itself carries no leak
independent of whether the original pipeline design is sound.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

DATA_PATH = Path(__file__).resolve().parent.parent / "research_datasets" / "old" / "psyche_d" / "anon_processed_df.parquet"
OUT_PATH = Path(__file__).resolve().parent / "audit_psyche_d_baseline_results_part_b.json"
RANDOM_STATE = 42
N_SPLITS = 5

SCREENER_COLS = [
    "sex", "race_white", "race_black", "race_hispanic", "race_asian", "race_other", "birthyear",
    "educ", "height", "weight", "bmi", "pregnant", "birth", "trauma", "insurance", "money",
    "money_assistance", "household", "comorbid_cancer", "comorbid_diabetes_typ1", "comorbid_diabetes_typ2",
    "comorbid_gout", "comorbid_migraines", "comorbid_ms", "comorbid_osteoporosis", "comorbid_neuropathic",
    "comorbid_arthritis", "num_migraine_days", "meds_migraine",
]
# Verbatim from the original repo's combined_pipeline.py (PHASE_1_INPUT_COLS
# literal list, before the SCREENER_COLS union it does at runtime).
PHASE_1_SLEEP_STEPS_COLS_LITERAL = [
    "sleep__main_start_hour_adj__score", "sleep__main_start_hour_adj__intercept", "sleep__main_start_hour_adj__coeff",
    "sleep_main_start_hour_adj_median", "sleep_main_start_hour_adj_iqr", "sleep_main_start_hour_adj_range",
    "sleep__total_asleep_minutes__score", "sleep__total_asleep_minutes__intercept", "sleep__total_asleep_minutes__coeff",
    "sleep__awake__sum__score", "sleep__awake__sum__intercept", "sleep__awake__sum__coeff", "sleep__nap_count__score",
    "sleep__nap_count__intercept", "sleep__nap_count__coeff",
    "sleep__total_asleep_minutes__score_",
    "sleep__total_asleep_minutes__intercept_", "sleep__total_asleep_minutes__coeff_", "sleep__main_efficiency__score_",
    "sleep__main_efficiency__intercept_", "sleep__main_efficiency__coeff_",
    "sleep__awake__sum__score_",
    "sleep__awake__sum__intercept_", "sleep__awake__sum__coeff_",
    "sleep__total_in_bed_minutes__score_",
    "sleep__total_in_bed_minutes__intercept_", "sleep__total_in_bed_minutes__coeff_",
    "steps__awake__sum__score_",
    "steps__awake__sum__intercept_", "steps__awake__sum__coeff_", "steps__mvpa__sum__score_",
    "steps__mvpa__sum__intercept_", "steps__mvpa__sum__coeff_", "steps__light_activity__sum__score_",
    "steps__light_activity__sum__intercept_", "steps__light_activity__sum__coeff_",
    "steps_mvpa_sum_recent",
    "sleep_asleep_mean_recent", "sleep_in_bed_mean_recent", "sleep_ratio_asleep_in_bed_mean_recent",
    "steps_lpa_sum_recent",
]
PROBA_COLS = ["proba_cat_0", "proba_cat_1", "proba_cat_2", "proba_cat_3", "proba_cat_4"]
PROBA_COLS_M1 = [c + "_m1" for c in PROBA_COLS]
PROBA_COLS_M2 = [c + "_m2" for c in PROBA_COLS]
GENERATED_COLS = ["cat_m0", "cat_m1", "cat_m2"] + PROBA_COLS_M1 + PROBA_COLS_M2
LEAKAGE_COLS = ["phq9_score_start", "phq9_score_end", "phq9_cat_start", "phq9_cat_end"]


def load_raw() -> pd.DataFrame:
    raw = pd.read_parquet(DATA_PATH)
    raw["user_id"] = [x.rsplit("_")[0] for x in raw.index]
    raw["user_mth"] = [int(x.rsplit("_")[1]) for x in raw.index]
    raw["ref_label"] = ~raw["phq9_cat_end"].isna()
    assert int(raw["ref_label"].sum()) == 10866, f"expected 10866 labeled rows, got {int(raw['ref_label'].sum())}"
    return raw


def build_phase1_input_cols(raw: pd.DataFrame) -> list[str]:
    lmc_cols = [x for x in raw.columns if x.startswith("life") or x.startswith("med") or x.startswith("nonmed")]
    cols = sorted(set(SCREENER_COLS + PHASE_1_SLEEP_STEPS_COLS_LITERAL + lmc_cols))
    missing = [c for c in cols if c not in raw.columns]
    assert not missing, f"PHASE_1_INPUT_COLS references columns absent from our parquet: {missing}"
    return cols


def build_phase2_generated_features(raw: pd.DataFrame, fold_labeled_idx: pd.Index, cat_end_col: pd.Series, proba_df: pd.DataFrame) -> pd.DataFrame:
    """For each labeled (quarter-end) row, look up its two preceding
    intermediate months (m-2, m-1) for the SAME user_id, and pull that
    month's phq9_cat_end value (always generated - m-2/m-1 are never
    quarter boundaries {3,6,9,12}, so always intermediate/unlabeled) plus
    its 5-way class-probability vector. A missing preceding month (no row
    for that exact month for that participant) -> NaN, left for the shared
    SimpleImputer(mean) to handle downstream, this project's standing
    convention."""
    out = pd.DataFrame(index=fold_labeled_idx, columns=GENERATED_COLS, dtype=np.float64)
    sub = raw.loc[:, ["user_id", "user_mth"]].copy()
    sub["cat_end_val"] = cat_end_col
    for c in PROBA_COLS:
        sub[c] = proba_df[c]
    sub_by_user = {u: g for u, g in sub.groupby("user_id")}
    cat_start_lookup = raw.loc[fold_labeled_idx, "phq9_cat_start"]
    user_lookup = raw.loc[fold_labeled_idx, "user_id"]
    mth_lookup = raw.loc[fold_labeled_idx, "user_mth"]
    for idx in fold_labeled_idx:
        u = user_lookup.loc[idx]
        m = int(mth_lookup.loc[idx])
        out.loc[idx, "cat_m0"] = cat_start_lookup.loc[idx]
        g = sub_by_user.get(u)
        if g is None:
            continue
        row_m2 = g[g["user_mth"] == m - 2]
        if len(row_m2) == 1:
            out.loc[idx, "cat_m1"] = row_m2["cat_end_val"].iloc[0]
            out.loc[idx, PROBA_COLS_M1] = row_m2[PROBA_COLS].iloc[0].to_numpy()
        row_m1 = g[g["user_mth"] == m - 1]
        if len(row_m1) == 1:
            out.loc[idx, "cat_m2"] = row_m1["cat_end_val"].iloc[0]
            out.loc[idx, PROBA_COLS_M2] = row_m1[PROBA_COLS].iloc[0].to_numpy()
    return out


def phase1_dart_factory():
    return lgb.LGBMClassifier(boosting_type="dart", n_estimators=100, max_depth=4, drop_rate=0.1, random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1)


def phase2_dart_factory():
    return lgb.LGBMClassifier(boosting_type="dart", n_estimators=150, max_depth=4, drop_rate=0.1, random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1)


def summarize(name: str, oof: np.ndarray, y: np.ndarray) -> dict:
    auroc = roc_auc_score(y, oof)
    auprc = average_precision_score(y, oof)
    predicted = (oof >= 0.5).astype(int)
    tp = int(((y == 1) & (predicted == 1)).sum())
    fp = int(((y == 0) & (predicted == 1)).sum())
    tn = int(((y == 0) & (predicted == 0)).sum())
    fn = int(((y == 1) & (predicted == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    acc = (tp + tn) / len(y)
    print(f"{name:50s} AUROC={auroc:.4f} AUPRC={auprc:.4f} acc={acc:.4f} sens={sens:.4f} spec={spec:.4f} TP/FP/TN/FN={tp}/{fp}/{tn}/{fn}")
    return {"auroc": float(auroc), "auprc": float(auprc), "accuracy": float(acc), "sensitivity": float(sens),
            "specificity": float(spec), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def run_reproduction(permutation_check: bool = False) -> dict:
    raw = load_raw()
    phase1_input_cols = build_phase1_input_cols(raw)
    print(f"PHASE_1_INPUT_COLS: {len(phase1_input_cols)} cols (all present in parquet)")

    phase2_base_cols = [c for c in raw.columns if c not in LEAKAGE_COLS and c not in ("user_id", "user_mth", "ref_label")]
    assert len(phase2_base_cols) == 150

    labeled = raw[raw["ref_label"]].copy()
    participant_ids = labeled["user_id"].to_numpy()
    delta_cat = labeled["phq9_cat_end"] - labeled["phq9_cat_start"]
    y_real = (delta_cat > 0).astype(int).to_numpy()

    if permutation_check:
        rng = np.random.default_rng(777)
        y = rng.permutation(y_real)
        print("*** PERMUTATION SANITY CHECK MODE: binary `worsened` label shuffled before fitting/scoring ***")
    else:
        y = y_real
    y_series = pd.Series(y, index=labeled.index)

    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    n = len(y)
    oof_with_intermediate = np.full(n, np.nan)
    oof_without_intermediate = np.full(n, np.nan)
    phase1_fold_metrics = []
    coverage_stats = []

    for fold_idx, (tr_pos, te_pos) in enumerate(skf.split(np.zeros(n), y, groups=participant_ids)):
        train_idx = labeled.index[tr_pos]
        test_idx = labeled.index[te_pos]
        train_participants = set(participant_ids[tr_pos])
        test_participants = set(participant_ids[te_pos])
        assert not (train_participants & test_participants), f"participant leakage fold {fold_idx}"

        # --- Phase 1: trained ONLY on this fold's train-participant labeled
        # rows, using the REAL 5-class phq9_cat_end (never the shuffled
        # binary label - Phase 1 doesn't see the binary target at all). ---
        phase1_train_mask = raw["ref_label"] & raw["user_id"].isin(train_participants)
        X1_train_df = raw.loc[phase1_train_mask, phase1_input_cols]
        y1_train = raw.loc[phase1_train_mask, "phq9_cat_end"].astype(int)

        imputer1 = SimpleImputer(strategy="mean", keep_empty_features=True)
        X1_train = imputer1.fit_transform(X1_train_df)
        sw1 = compute_sample_weight(class_weight="balanced", y=y1_train)
        clf1 = phase1_dart_factory()
        clf1.fit(X1_train, y1_train, sample_weight=sw1)

        X1_test = imputer1.transform(raw.loc[test_idx, phase1_input_cols])
        y1_test = raw.loc[test_idx, "phq9_cat_end"].astype(int)
        y1_pred = clf1.predict(X1_test)
        phase1_fold_metrics.append({
            "fold": fold_idx + 1,
            "n_phase1_train_rows": int(phase1_train_mask.sum()),
            "accuracy": float(accuracy_score(y1_test, y1_pred)),
            "adjacent_accuracy": float(np.mean(np.abs(y1_pred - y1_test.to_numpy()) <= 1)),
            "balanced_accuracy": float(balanced_accuracy_score(y1_test, y1_pred)),
            "kappa": float(cohen_kappa_score(y1_test, y1_pred, weights="quadratic")),
        })

        # --- Generate intermediate-month labels for ALL unlabeled rows
        # (train-fold AND test-fold participants' intermediate months)
        # using the fold's train-only Phase-1 model. Leakage-safe:
        # inference only, never uses any test participant's true label. ---
        unlabeled_mask = ~raw["ref_label"]
        Xg = imputer1.transform(raw.loc[unlabeled_mask, phase1_input_cols])
        gen_proba = clf1.predict_proba(Xg)
        gen_pred = clf1.classes_[np.argmax(gen_proba, axis=1)]
        cat_end_generated = raw["phq9_cat_end"].copy()
        cat_end_generated.loc[unlabeled_mask] = gen_pred
        proba_df = pd.DataFrame(0.0, index=raw.index, columns=PROBA_COLS)
        for i, c in enumerate(clf1.classes_):
            proba_df.loc[unlabeled_mask, PROBA_COLS[int(c)]] = gen_proba[:, i]

        fold_labeled_idx = train_idx.append(test_idx)
        gen_feats = build_phase2_generated_features(raw, fold_labeled_idx, cat_end_generated, proba_df)
        coverage_stats.append({
            "fold": fold_idx + 1,
            "n_generated_rows": int(unlabeled_mask.sum()),
            "frac_cat_m1_available": float(gen_feats["cat_m1"].notna().mean()),
            "frac_cat_m2_available": float(gen_feats["cat_m2"].notna().mean()),
        })

        X2_base = raw.loc[fold_labeled_idx, phase2_base_cols]
        X2_full = pd.concat([X2_base, gen_feats], axis=1)
        y2 = y_series.loc[fold_labeled_idx]

        tr_mask2 = X2_full.index.isin(train_idx)
        te_mask2 = X2_full.index.isin(test_idx)

        # Phase 2 WITH generated intermediate features (the reproduction arm)
        imputer2 = SimpleImputer(strategy="mean", keep_empty_features=True)
        X2_train = imputer2.fit_transform(X2_full.loc[tr_mask2])
        X2_test = imputer2.transform(X2_full.loc[te_mask2])
        sw2 = compute_sample_weight(class_weight="balanced", y=y2.loc[tr_mask2])
        clf2 = phase2_dart_factory()
        clf2.fit(X2_train, y2.loc[tr_mask2], sample_weight=sw2)
        oof_with_intermediate[te_pos] = clf2.predict_proba(X2_test)[:, 1]

        # Phase 2 WITHOUT generated intermediate features (ablation, Part C:
        # isolates whether any gain is specifically from the intermediate-
        # label mechanism vs. just LightGBM-DART + more estimators/depth)
        imputer2b = SimpleImputer(strategy="mean", keep_empty_features=True)
        X2b_train = imputer2b.fit_transform(X2_base.loc[tr_mask2])
        X2b_test = imputer2b.transform(X2_base.loc[te_mask2])
        clf2b = phase2_dart_factory()
        clf2b.fit(X2b_train, y2.loc[tr_mask2], sample_weight=sw2)
        oof_without_intermediate[te_pos] = clf2b.predict_proba(X2b_test)[:, 1]

        print(f"fold {fold_idx + 1}: phase1_train_rows={int(phase1_train_mask.sum())} "
              f"phase1_acc={phase1_fold_metrics[-1]['accuracy']:.4f} "
              f"cat_m1_avail={coverage_stats[-1]['frac_cat_m1_available']:.3f} "
              f"cat_m2_avail={coverage_stats[-1]['frac_cat_m2_available']:.3f}")

    assert not np.isnan(oof_with_intermediate).any()
    assert not np.isnan(oof_without_intermediate).any()

    tag = "_PERMUTED" if permutation_check else ""
    results = {
        "permutation_check": permutation_check,
        "phase1_input_cols": phase1_input_cols,
        "phase1_fold_metrics": phase1_fold_metrics,
        "coverage_stats": coverage_stats,
        f"phase2_with_intermediate{tag}": summarize(f"Phase2 LightGBM-DART + generated intermediate{tag}", oof_with_intermediate, y),
        f"phase2_without_intermediate{tag}": summarize(f"Phase2 LightGBM-DART, no intermediate (ablation){tag}", oof_without_intermediate, y),
        "phase1_mean_accuracy": float(np.mean([f["accuracy"] for f in phase1_fold_metrics])),
        "phase1_mean_adjacent_accuracy": float(np.mean([f["adjacent_accuracy"] for f in phase1_fold_metrics])),
        "phase1_mean_kappa": float(np.mean([f["kappa"] for f in phase1_fold_metrics])),
    }
    print(f"\nPhase 1 (5-class phq9_cat_end prediction) pooled-mean: "
          f"accuracy={results['phase1_mean_accuracy']:.4f} "
          f"adjacent_accuracy={results['phase1_mean_adjacent_accuracy']:.4f} "
          f"quadratic_kappa={results['phase1_mean_kappa']:.4f}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutation-check", action="store_true", help="Run the label-shuffle sanity check instead of the real reproduction.")
    args = parser.parse_args()

    print(f"pandas={pd.__version__} numpy={np.__version__} lightgbm={lgb.__version__}")
    results = run_reproduction(permutation_check=args.permutation_check)

    out_path = OUT_PATH.with_name(OUT_PATH.stem + ("_permutation_check" if args.permutation_check else "") + ".json")
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
