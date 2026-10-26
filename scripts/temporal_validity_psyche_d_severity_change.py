#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), derived from the DiSCover Project
# (NCT03421223), CC BY-NC 4.0 — a separate institution's protocol,
# independent of this project's own ethics protocol (REB25-1587_MOD1).
# `research_datasets/` is gitignored; the parquet is local research data
# and is never committed to this repo.
"""PSYCHE-D temporal validity check + out-of-time diagnostic.

Investigation #2 of the `psyche-d-baseline-freeze` debug session
(`.planning/debug/psyche-d-baseline-freeze.md`). Focuses on the arm B
"~0.69 baseline" (150 base features + phq9_score_start/phq9_cat_start
baseline-severity covariates — the arm the "~0.69 AUROC" framing in this
session's task spec refers to), NOT the two-stage arm C (that arm's own
leakage question is Investigation #1, handled separately in
`crossfit_psyche_d_two_stage_pipeline.py`).

WHAT WAS ALREADY ESTABLISHED (prior audit, NOT repeated here — see
`.planning/debug/resolved/audit-psyche-d-baseline.md` and
`scripts/audit_psyche_d_baseline_results.json`):
  - No absolute calendar date/enrollment-date field exists anywhere in the
    154-column frame — only relative `user_mth` (1-12, quarter-end labels
    at {3,6,9,12}). Confirmed by this session's own evidence-gathering
    (grep across all columns), not re-derived here.
  - Worsened-rate is stable across quarters (~19.9%-22.1%, A12), and
    grouped-CV vs random-split AUROC on the 150-feature arm are nearly
    identical (0.5694 vs 0.5699, A11) — already rules out gross temporal
    drift or grouped-CV-vs-random-split sensitivity on the base arm.
  - `StratifiedGroupKFold(groups=user_id)` means a participant's rows are
    NEVER split across train/test within a fold — BY CONSTRUCTION, a
    participant's later observation can never sit in train while an
    earlier observation of theirs sits in test (or vice versa) under the
    standing protocol. Re-verified here as an in-code assertion, not
    re-argued from scratch.

WHAT'S NEW HERE: no calendar date exists, so a true calendar-based
out-of-time split is impossible. The only available temporal axis is each
participant's own relative `user_mth`. Two complementary out-of-time
diagnostics are run, single train/test splits (not k-fold — "perform ONE
out-of-time evaluation" per the task spec), on the IDENTICAL 152-column
(150 base + baseline severity) feature set and the IDENTICAL
`phase2_dart_factory()` model as this session's other new scripts, so
results are comparable to a same-model/same-features grouped-CV control
computed in this same script:

  1. ROW-LEVEL FORWARD CHAINING (`temporal_split_naive`). Train on all
     labeled rows at user_mth in {3,6} ("early"), test on all labeled rows
     at user_mth in {9,12} ("late"). A participant CAN legitimately appear
     in both sets (their own early quarter in train, their own late
     quarter in test) — this is intentional and different in kind from the
     grouped-CV convention, not a contamination bug; stated explicitly.
     Tests pure forward-in-time generalization of the row-level signal.

  2. PARTICIPANT+TEMPORAL DISJOINT (`temporal_split_disjoint`). Test
     participants = those with a labeled row in {9,12} AND NO labeled row
     in {3,6} ("late-only" participants) — held out ENTIRELY from train
     (zero participant overlap, verified by assertion). Train = every
     other participant's labeled row(s), any quarter. Closest analog to
     real deployment: predicting worsening for a participant only observed
     later in the study, using a model that never saw them at all.

A `grouped_cv_same_model_control` arm (standard `StratifiedGroupKFold(5)`,
identical features/factory) is also computed in this script so any gap
between it and the two out-of-time splits is not confounded by a different
model configuration than the previously-cited B6 arm (which used
`n_estimators=300, is_unbalance=True` — a different factory than this
session's `REPRO.phase2_dart_factory()`).

REUSE, NOT REIMPLEMENTATION. Imports (never copies) `load_raw`, `LEAKAGE_COLS`,
`RANDOM_STATE`, `N_SPLITS`, `phase2_dart_factory`, `summarize` from
`audit_psyche_d_baseline_reproduction` (`REPRO`); `bootstrap_ci`,
`N_BOOTSTRAP` from `train_psyche_d_severity_change`. Does not modify any
existing `scripts/*psyche_d*` file in place.

No participant ID, per-participant row, or per-row export is ever written to
the results JSON or report — aggregate metrics only.
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_psyche_d_baseline_reproduction as REPRO  # noqa: E402
from train_psyche_d_severity_change import bootstrap_ci, N_BOOTSTRAP  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "temporal_validity_psyche_d_severity_change_results.json"
REPORT_PATH = SCRIPT_DIR / "temporal_validity_psyche_d_severity_change_report.md"

BASELINE_SEVERITY_COLS = ["phq9_score_start", "phq9_cat_start"]
EARLY_QUARTERS = {3, 6}
LATE_QUARTERS = {9, 12}


def evaluate_split(X_train_df, y_train, X_test_df, y_test, label: str) -> dict:
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    X_train = imputer.fit_transform(X_train_df)
    X_test = imputer.transform(X_test_df)
    sw = compute_sample_weight(class_weight="balanced", y=y_train)
    clf = REPRO.phase2_dart_factory()
    clf.fit(X_train, y_train, sample_weight=sw)
    proba = clf.predict_proba(X_test)[:, 1]
    summary = REPRO.summarize(label, proba, y_test)
    auroc_lo, auroc_hi, _ = bootstrap_ci(y_test, proba, roc_auc_score)
    auprc_lo, auprc_hi, _ = bootstrap_ci(y_test, proba, average_precision_score)
    summary["auroc_ci"] = [auroc_lo, auroc_hi]
    summary["auprc_ci"] = [auprc_lo, auprc_hi]
    summary["n_train"] = int(len(y_train))
    summary["n_test"] = int(len(y_test))
    summary["train_positive_rate"] = float(np.mean(y_train))
    summary["test_positive_rate"] = float(np.mean(y_test))
    return summary


def run() -> dict:
    t0 = time.time()
    raw = REPRO.load_raw()
    phase2_base_cols = [c for c in raw.columns if c not in REPRO.LEAKAGE_COLS and c not in ("user_id", "user_mth", "ref_label")]
    assert len(phase2_base_cols) == 150
    feature_cols = phase2_base_cols + BASELINE_SEVERITY_COLS
    assert len(feature_cols) == 152

    labeled = raw[raw["ref_label"]].copy()
    delta_cat = labeled["phq9_cat_end"] - labeled["phq9_cat_start"]
    labeled["y"] = (delta_cat > 0).astype(int)
    participant_ids = labeled["user_id"].to_numpy()

    # --- Repeated-observations / grouped-CV-sufficiency re-check: verify by
    # assertion (not re-argued) that StratifiedGroupKFold never splits a
    # participant's rows across train/test within a fold. ---
    skf = StratifiedGroupKFold(n_splits=REPRO.N_SPLITS, shuffle=True, random_state=REPRO.RANDOM_STATE)
    y_all = labeled["y"].to_numpy()
    n = len(y_all)
    for tr_pos, te_pos in skf.split(np.zeros(n), y_all, groups=participant_ids):
        assert not (set(participant_ids[tr_pos]) & set(participant_ids[te_pos])), (
            "grouped CV split a participant's rows across train/test — this would mean a "
            "participant's later observation could influence prediction of their own earlier "
            "observation (or vice versa) within a fold. NOT observed."
        )
    print("Grouped-CV participant-split assertion: PASSED for all 5 folds (no participant ever split across train/test within a fold).")

    # --- grouped_cv_same_model_control: standard protocol, same features/factory as the two out-of-time arms below, for a fair comparison. ---
    oof_grouped = np.full(n, np.nan)
    for tr_pos, te_pos in skf.split(np.zeros(n), y_all, groups=participant_ids):
        X_tr_df = labeled.iloc[tr_pos][feature_cols]
        X_te_df = labeled.iloc[te_pos][feature_cols]
        imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
        X_tr = imputer.fit_transform(X_tr_df)
        X_te = imputer.transform(X_te_df)
        sw = compute_sample_weight(class_weight="balanced", y=y_all[tr_pos])
        clf = REPRO.phase2_dart_factory()
        clf.fit(X_tr, y_all[tr_pos], sample_weight=sw)
        oof_grouped[te_pos] = clf.predict_proba(X_te)[:, 1]
    assert not np.isnan(oof_grouped).any()
    grouped_summary = REPRO.summarize("grouped_cv_same_model_control (152 feat)", oof_grouped, y_all)
    g_lo, g_hi, _ = bootstrap_ci(y_all, oof_grouped, roc_auc_score)
    ga_lo, ga_hi, _ = bootstrap_ci(y_all, oof_grouped, average_precision_score)
    grouped_summary["auroc_ci"] = [g_lo, g_hi]
    grouped_summary["auprc_ci"] = [ga_lo, ga_hi]

    # --- Split 1: row-level forward chaining (naive, participant overlap allowed) ---
    early_mask = labeled["user_mth"].isin(EARLY_QUARTERS)
    late_mask = labeled["user_mth"].isin(LATE_QUARTERS)
    assert (early_mask | late_mask).all(), "every labeled row must be at a quarter boundary {3,6,9,12}"
    n_overlap_participants = len(set(labeled.loc[early_mask, "user_id"]) & set(labeled.loc[late_mask, "user_id"]))
    naive_summary = evaluate_split(
        labeled.loc[early_mask, feature_cols], labeled.loc[early_mask, "y"].to_numpy(),
        labeled.loc[late_mask, feature_cols], labeled.loc[late_mask, "y"].to_numpy(),
        "temporal_split_naive (train early {3,6} -> test late {9,12})",
    )
    naive_summary["n_participants_appearing_in_both_train_and_test"] = n_overlap_participants

    # --- Split 2: participant+temporal disjoint ---
    late_only_participants = set(labeled.loc[late_mask, "user_id"]) - set(labeled.loc[early_mask, "user_id"])
    test_mask_disjoint = labeled["user_id"].isin(late_only_participants) & late_mask
    train_mask_disjoint = ~labeled["user_id"].isin(late_only_participants)
    assert not (set(labeled.loc[train_mask_disjoint, "user_id"]) & set(labeled.loc[test_mask_disjoint, "user_id"])), \
        "participant+temporal disjoint split must have zero participant overlap"
    disjoint_summary = evaluate_split(
        labeled.loc[train_mask_disjoint, feature_cols], labeled.loc[train_mask_disjoint, "y"].to_numpy(),
        labeled.loc[test_mask_disjoint, feature_cols], labeled.loc[test_mask_disjoint, "y"].to_numpy(),
        "temporal_split_disjoint (train = everyone else, test = late-only unseen participants)",
    )
    disjoint_summary["n_late_only_participants"] = len(late_only_participants)

    results = {
        "meta": {
            "script": "scripts/temporal_validity_psyche_d_severity_change.py",
            "debug_session": ".planning/debug/psyche-d-baseline-freeze.md",
            "feature_cols_count": len(feature_cols),
            "n_boot": N_BOOTSTRAP,
            "runtime_sec": time.time() - t0,
            "versions": {"pandas": pd.__version__, "numpy": np.__version__, "lightgbm": lgb.__version__, "sklearn": sklearn.__version__},
        },
        "grouped_cv_participant_split_assertion": "PASSED",
        "grouped_cv_same_model_control": grouped_summary,
        "temporal_split_naive": naive_summary,
        "temporal_split_disjoint": disjoint_summary,
        "worsened_rate_by_quarter_prior_finding": {
            "source": "scripts/audit_psyche_d_baseline_results.json:part_a.A12_worsen_rate_by_month",
            "3": 0.20468557336621454, "6": 0.22069209039548024, "9": 0.19948755490483164, "12": 0.2031098153547133,
        },
    }
    print(f"\ngrouped_cv_same_model_control  AUROC={grouped_summary['auroc']:.4f} CI={grouped_summary['auroc_ci']}")
    print(f"temporal_split_naive           AUROC={naive_summary['auroc']:.4f} CI={naive_summary['auroc_ci']} "
          f"(n_train={naive_summary['n_train']} n_test={naive_summary['n_test']} overlap_participants={n_overlap_participants})")
    print(f"temporal_split_disjoint        AUROC={disjoint_summary['auroc']:.4f} CI={disjoint_summary['auroc_ci']} "
          f"(n_train={disjoint_summary['n_train']} n_test={disjoint_summary['n_test']} late_only_participants={len(late_only_participants)})")
    return results


def write_report(results: dict) -> None:
    g = results["grouped_cv_same_model_control"]
    a = results["temporal_split_naive"]
    b = results["temporal_split_disjoint"]
    lines = [
        "# PSYCHE-D Investigation #2 — Temporal Validity Check + Out-of-Time Diagnostic",
        "",
        f"Debug session: `{results['meta']['debug_session']}`",
        f"Script: `{results['meta']['script']}` (152 features, n_boot={results['meta']['n_boot']}, runtime={results['meta']['runtime_sec']:.1f}s)",
        "",
        "## Pre-established (not repeated here)",
        "",
        "- No calendar date field exists — only relative `user_mth` (1-12).",
        "- Worsened-rate stable across quarters (~19.9%-22.1%).",
        "- Grouped-CV vs random-split AUROC nearly identical on the 150-feature arm (0.5694 vs 0.5699).",
        f"- Grouped-CV participant-split assertion: **{results['grouped_cv_participant_split_assertion']}** — a participant's rows are never split across train/test within a fold, by construction.",
        "",
        "## Results",
        "",
        "| Arm | AUROC | AUROC 95% CI | AUPRC | Sensitivity | Specificity | n_train | n_test |",
        "|---|---|---|---|---|---|---|---|",
        f"| grouped_cv_same_model_control | {g['auroc']:.4f} | [{g['auroc_ci'][0]:.4f}, {g['auroc_ci'][1]:.4f}] | {g['auprc']:.4f} | {g['sensitivity']:.4f} | {g['specificity']:.4f} | pooled OOF (10866) | pooled OOF |",
        f"| temporal_split_naive | {a['auroc']:.4f} | [{a['auroc_ci'][0]:.4f}, {a['auroc_ci'][1]:.4f}] | {a['auprc']:.4f} | {a['sensitivity']:.4f} | {a['specificity']:.4f} | {a['n_train']} | {a['n_test']} |",
        f"| temporal_split_disjoint | {b['auroc']:.4f} | [{b['auroc_ci'][0]:.4f}, {b['auroc_ci'][1]:.4f}] | {b['auprc']:.4f} | {b['sensitivity']:.4f} | {b['specificity']:.4f} | {b['n_train']} | {b['n_test']} |",
        "",
        f"temporal_split_naive: {a['n_participants_appearing_in_both_train_and_test']} participants appear in BOTH train (early quarter) and test (late quarter) — intentional for this split.",
        f"temporal_split_disjoint: {b['n_late_only_participants']} late-only participants held out entirely as test — zero participant overlap with train (asserted).",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote {REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(f"pandas={pd.__version__} numpy={np.__version__} lightgbm={lgb.__version__} sklearn={sklearn.__version__}")
    results = run()
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {RESULTS_PATH}")
    write_report(results)


if __name__ == "__main__":
    main()
