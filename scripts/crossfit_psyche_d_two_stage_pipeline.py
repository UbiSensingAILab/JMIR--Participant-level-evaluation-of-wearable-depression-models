#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), derived from the DiSCover Project
# (NCT03421223), CC BY-NC 4.0 — a separate institution's protocol,
# independent of this project's own ethics protocol (REB25-1587_MOD1).
# `research_datasets/` is gitignored; the parquet is local research data
# and is never committed to this repo.
"""PSYCHE-D two-stage pipeline — participant-level cross-fitted Phase 1
generation for TRAIN-side rows.

Investigation #1 (highest priority) of the `psyche-d-baseline-freeze` debug
session (`.planning/debug/psyche-d-baseline-freeze.md`).

THE GAP THIS FIXES. `audit_psyche_d_baseline_reproduction.py` was verified
leakage-safe for the EVAL side only (a genuinely held-out fold-test
participant's intermediate features never come from a Phase-1 model that saw
their own quarter-end label). It was never checked for the TRAIN side: per
outer fold, ONE `clf1` is fit on every outer-TRAIN participant's labeled
row(s), then used via `.predict_proba()` inference to generate intermediate
(m-1/m-2) features for those SAME participants' own rows, which become
Phase-2 TRAINING features. For a participant with 2+ labeled quarters, `clf1`
was fit on that exact participant's OTHER quarter-end label(s) while
generating features later used to predict one of their own quarters — the
generated intermediate feature is in-sample with respect to Phase 1, even
though Phase 2 itself never sees the raw label directly. Confirmed by direct
code trace (`audit_psyche_d_baseline_reproduction.py:243-322`) — see the
debug session's Evidence section for the full trace.

THE FIX. Participant-level cross-fitting, applied ONLY to outer-TRAIN
participants (outer-TEST participants already use the correct, test-time-
equivalent path and are left unchanged):

  - `clf1_full` — fit once per outer fold on ALL outer-TRAIN participants'
    labeled rows, IDENTICAL to REPRO. Used (unchanged) to generate outer-TEST
    participants' intermediate features via inference only. This path is
    already leakage-safe and is not touched.
  - Inner `KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=...)` over
    the outer fold's TRAIN participants (participant-level, not row-level).
    For each inner fold j: fit a fresh `clf1_inner_j` (own imputer AND own
    LightGBM-DART fit) on the OTHER inner-fold participants' labeled rows
    only — i.e. explicitly EXCLUDING inner-fold-j's own participants, so
    the imputer's per-column means and the tree splits never see anything
    about them, not just their label. Generate inner-fold-j's participants'
    intermediate features via inference only. Across all INNER_SPLITS inner
    folds, every outer-TRAIN participant gets exactly one cross-fitted
    generation pass, from a model that never trained on them.
  - Phase 2 is then trained on the cross-fitted TRAIN-side features and
    evaluated on the unchanged clf1_full-based TEST-side features — same
    `phase2_dart_factory()` hyperparameters as REPRO, so any measured
    difference is attributable to the Phase-1 generation mechanism, not a
    Phase-2 model change.

Both the CROSS-FITTED arm and the ORIGINAL in-sample arm (arm C) are computed
in THIS SAME SCRIPT, on the IDENTICAL outer `StratifiedGroupKFold` fold split
(same `RANDOM_STATE`/`N_SPLITS`, same call pattern as REPRO), so the two
pooled-OOF probability vectors are directly, row-for-row paired — enabling a
valid paired-bootstrap AUROC/AUPRC delta, not just two independent CIs. The
original arm's own pooled AUROC is asserted against the cited reference
0.6529684832943685 (regression guard — if this doesn't reconcile, the fold
reproduction itself is wrong and the cross-fit comparison is not trustworthy).

REUSE, NOT REIMPLEMENTATION. Following this repo's established cross-script
convention, this script imports (never copies) from
`audit_psyche_d_baseline_reproduction` (`REPRO`): `load_raw`,
`build_phase1_input_cols`, `build_phase2_generated_features`,
`phase1_dart_factory`, `phase2_dart_factory`, `summarize`, `PROBA_COLS`,
`PROBA_COLS_M1`, `PROBA_COLS_M2`, `GENERATED_COLS`, `LEAKAGE_COLS`,
`RANDOM_STATE`, `N_SPLITS`; `bootstrap_ci`, `N_BOOTSTRAP` from
`train_psyche_d_severity_change`; and `paired_bootstrap_auroc_delta` from
`tune_psyche_d_severity_change_models`. `audit_psyche_d_baseline_reproduction.py`
is NOT modified in place — this is a new script alongside it.

SCOPE NOTE: the trailing `--permutation-check` sanity check that
`audit_psyche_d_baseline_reproduction.py` offers is intentionally NOT
repeated here — REPRO's own permutation check already established the
reproduction pipeline's mechanics carry no leak independent of the Phase-1
generation design being investigated here, and this session's explicit
non-negotiable constraint is "no broad hyperparameter optimization" /
"do not repeat the prior audit's already-settled scope." Cross-fitting is a
generation-mechanism fix, not a new leak-safety class requiring its own
permutation check.

No participant ID, per-participant row, or per-row export is ever written to
the results JSON or report — aggregate metrics only, per this dataset's
pre-registered no-per-participant-export rule.
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
from sklearn.model_selection import KFold, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_psyche_d_baseline_reproduction as REPRO  # noqa: E402
from train_psyche_d_severity_change import bootstrap_ci, N_BOOTSTRAP  # noqa: E402
from tune_psyche_d_severity_change_models import paired_bootstrap_auroc_delta  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "crossfit_psyche_d_two_stage_pipeline_results.json"
REPORT_PATH = SCRIPT_DIR / "crossfit_psyche_d_two_stage_pipeline_report.md"

INNER_SPLITS = 5

# Cited verbatim from scripts/audit_psyche_d_baseline_results.json,
# key part_b_reproduction.phase2_with_intermediate — the regression-guard
# reconciliation target for the ORIGINAL (in-sample) arm reconstructed here.
CITED_REFERENCE = {
    "auroc": 0.6529684832943685,
    "accuracy": 0.6175225473955457,
    "sensitivity": 0.5954706927175843,
    "specificity": 0.6232876712328768,
    "tp": 1341, "fp": 3245, "tn": 5369, "fn": 911,
    "source": "scripts/audit_psyche_d_baseline_results.json:part_b_reproduction.phase2_with_intermediate",
}
RECONCILIATION_TOLERANCE = 0.002


def fit_phase1(raw: pd.DataFrame, phase1_input_cols: list[str], train_mask: pd.Series):
    """Fit a fresh imputer + LightGBM-DART Phase-1 model on exactly the rows
    selected by train_mask. Returns (imputer, clf) — a self-contained unit
    that has seen nothing outside train_mask, not even for imputation means."""
    X_df = raw.loc[train_mask, phase1_input_cols]
    y = raw.loc[train_mask, "phq9_cat_end"].astype(int)
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    X = imputer.fit_transform(X_df)
    sw = compute_sample_weight(class_weight="balanced", y=y)
    clf = REPRO.phase1_dart_factory()
    clf.fit(X, y, sample_weight=sw)
    return imputer, clf


def generate_intermediate(raw: pd.DataFrame, phase1_input_cols: list[str], imputer, clf, target_mask: pd.Series,
                           cat_end_out: pd.Series, proba_out: pd.DataFrame) -> None:
    """Inference-only: generate intermediate-month cat/proba predictions for
    rows selected by target_mask, writing into cat_end_out/proba_out
    in place. Never touches target_mask rows' true labels."""
    if not target_mask.any():
        return
    Xg = imputer.transform(raw.loc[target_mask, phase1_input_cols])
    gen_proba = clf.predict_proba(Xg)
    gen_pred = clf.classes_[np.argmax(gen_proba, axis=1)]
    cat_end_out.loc[target_mask] = gen_pred
    for i, c in enumerate(clf.classes_):
        proba_out.loc[target_mask, REPRO.PROBA_COLS[int(c)]] = gen_proba[:, i]


def run(inner_splits: int = INNER_SPLITS) -> dict:
    raw = REPRO.load_raw()
    phase1_input_cols = REPRO.build_phase1_input_cols(raw)
    phase2_base_cols = [c for c in raw.columns if c not in REPRO.LEAKAGE_COLS and c not in ("user_id", "user_mth", "ref_label")]
    assert len(phase2_base_cols) == 150

    labeled = raw[raw["ref_label"]].copy()
    participant_ids = labeled["user_id"].to_numpy()
    delta_cat = labeled["phq9_cat_end"] - labeled["phq9_cat_start"]
    y_real = (delta_cat > 0).astype(int).to_numpy()
    y_series = pd.Series(y_real, index=labeled.index)
    n = len(y_real)

    skf = StratifiedGroupKFold(n_splits=REPRO.N_SPLITS, shuffle=True, random_state=REPRO.RANDOM_STATE)

    oof_original = np.full(n, np.nan)
    oof_crossfit = np.full(n, np.nan)
    phase1_fold_metrics_full = []
    phase1_fold_metrics_inner = []
    fold_coverage = []

    t0 = time.time()
    for fold_idx, (tr_pos, te_pos) in enumerate(skf.split(np.zeros(n), y_real, groups=participant_ids)):
        train_idx = labeled.index[tr_pos]
        test_idx = labeled.index[te_pos]
        train_participants = sorted(set(participant_ids[tr_pos]))
        test_participants = set(participant_ids[te_pos])
        assert not (set(train_participants) & test_participants), f"participant leakage fold {fold_idx}"

        # --- clf1_full: identical to REPRO. Used for (a) the ORIGINAL arm's
        # train-side generation (in-sample, reproducing arm C exactly) and
        # (b) BOTH arms' test-side generation (already correct, test-time-
        # equivalent, unchanged). ---
        phase1_train_mask_full = raw["ref_label"] & raw["user_id"].isin(train_participants)
        imputer1_full, clf1_full = fit_phase1(raw, phase1_input_cols, phase1_train_mask_full)

        X1_test = imputer1_full.transform(raw.loc[test_idx, phase1_input_cols])
        y1_test = raw.loc[test_idx, "phq9_cat_end"].astype(int)
        y1_pred_full = clf1_full.predict(X1_test)
        phase1_fold_metrics_full.append({
            "fold": fold_idx + 1,
            "n_phase1_train_rows": int(phase1_train_mask_full.sum()),
            "accuracy": float((y1_pred_full == y1_test.to_numpy()).mean()),
        })

        unlabeled_mask = ~raw["ref_label"]
        test_participant_unlabeled_mask = unlabeled_mask & raw["user_id"].isin(test_participants)
        train_participant_unlabeled_mask = unlabeled_mask & raw["user_id"].isin(train_participants)

        # ===== ORIGINAL (in-sample) arm — reconstructs arm C exactly =====
        cat_end_original = raw["phq9_cat_end"].copy()
        proba_original = pd.DataFrame(0.0, index=raw.index, columns=REPRO.PROBA_COLS)
        generate_intermediate(raw, phase1_input_cols, imputer1_full, clf1_full, unlabeled_mask, cat_end_original, proba_original)

        fold_labeled_idx = train_idx.append(test_idx)
        gen_feats_original = REPRO.build_phase2_generated_features(raw, fold_labeled_idx, cat_end_original, proba_original)

        # ===== CROSS-FITTED arm — TEST side unchanged (clf1_full), TRAIN
        # side from inner-fold models that excluded each participant =====
        cat_end_crossfit = raw["phq9_cat_end"].copy()
        proba_crossfit = pd.DataFrame(0.0, index=raw.index, columns=REPRO.PROBA_COLS)
        # Test side: identical path to the original arm (already leakage-safe).
        generate_intermediate(raw, phase1_input_cols, imputer1_full, clf1_full, test_participant_unlabeled_mask, cat_end_crossfit, proba_crossfit)

        inner_kf = KFold(n_splits=inner_splits, shuffle=True, random_state=REPRO.RANDOM_STATE * 10_000 + fold_idx)
        train_participants_arr = np.array(train_participants)
        covered = np.zeros(len(train_participants_arr), dtype=bool)
        for inner_idx, (inner_tr_pos, inner_ho_pos) in enumerate(inner_kf.split(train_participants_arr)):
            inner_train_participants = set(train_participants_arr[inner_tr_pos])
            inner_holdout_participants = set(train_participants_arr[inner_ho_pos])
            assert not (inner_train_participants & inner_holdout_participants)
            covered[inner_ho_pos] = True

            inner_train_mask = raw["ref_label"] & raw["user_id"].isin(inner_train_participants)
            imputer1_inner, clf1_inner = fit_phase1(raw, phase1_input_cols, inner_train_mask)

            inner_holdout_labeled_idx = labeled.index[labeled["user_id"].isin(inner_holdout_participants)]
            if len(inner_holdout_labeled_idx) > 0:
                y1_inner_test = raw.loc[inner_holdout_labeled_idx, "phq9_cat_end"].astype(int)
                X1_inner_test = imputer1_inner.transform(raw.loc[inner_holdout_labeled_idx, phase1_input_cols])
                y1_inner_pred = clf1_inner.predict(X1_inner_test)
                phase1_fold_metrics_inner.append({
                    "outer_fold": fold_idx + 1,
                    "inner_fold": inner_idx + 1,
                    "n_phase1_train_rows": int(inner_train_mask.sum()),
                    "n_holdout_participants": len(inner_holdout_participants),
                    "accuracy": float((y1_inner_pred == y1_inner_test.to_numpy()).mean()),
                })

            inner_holdout_unlabeled_mask = unlabeled_mask & raw["user_id"].isin(inner_holdout_participants)
            generate_intermediate(raw, phase1_input_cols, imputer1_inner, clf1_inner, inner_holdout_unlabeled_mask, cat_end_crossfit, proba_crossfit)

        assert covered.all(), f"fold {fold_idx}: inner KFold did not cover every outer-train participant exactly once"
        gen_feats_crossfit = REPRO.build_phase2_generated_features(raw, fold_labeled_idx, cat_end_crossfit, proba_crossfit)

        fold_coverage.append({
            "fold": fold_idx + 1,
            "n_train_participants": len(train_participants),
            "n_inner_folds": inner_splits,
            "frac_cat_m1_available_original": float(gen_feats_original["cat_m1"].notna().mean()),
            "frac_cat_m1_available_crossfit": float(gen_feats_crossfit["cat_m1"].notna().mean()),
        })

        tr_mask2 = fold_labeled_idx.isin(train_idx)
        te_mask2 = fold_labeled_idx.isin(test_idx)
        X2_base = raw.loc[fold_labeled_idx, phase2_base_cols]
        y2 = y_series.loc[fold_labeled_idx]
        sw2 = compute_sample_weight(class_weight="balanced", y=y2.loc[tr_mask2])

        # --- Fit/eval ORIGINAL arm ---
        X2_full_original = pd.concat([X2_base, gen_feats_original], axis=1)
        imputer2a = SimpleImputer(strategy="mean", keep_empty_features=True)
        X2a_train = imputer2a.fit_transform(X2_full_original.loc[tr_mask2])
        X2a_test = imputer2a.transform(X2_full_original.loc[te_mask2])
        clf2a = REPRO.phase2_dart_factory()
        clf2a.fit(X2a_train, y2.loc[tr_mask2], sample_weight=sw2)
        oof_original[te_pos] = clf2a.predict_proba(X2a_test)[:, 1]

        # --- Fit/eval CROSS-FITTED arm ---
        X2_full_crossfit = pd.concat([X2_base, gen_feats_crossfit], axis=1)
        imputer2b = SimpleImputer(strategy="mean", keep_empty_features=True)
        X2b_train = imputer2b.fit_transform(X2_full_crossfit.loc[tr_mask2])
        X2b_test = imputer2b.transform(X2_full_crossfit.loc[te_mask2])
        clf2b = REPRO.phase2_dart_factory()
        clf2b.fit(X2b_train, y2.loc[tr_mask2], sample_weight=sw2)
        oof_crossfit[te_pos] = clf2b.predict_proba(X2b_test)[:, 1]

        print(f"fold {fold_idx + 1}/{REPRO.N_SPLITS}: n_train_participants={len(train_participants)} "
              f"phase1_full_acc={phase1_fold_metrics_full[-1]['accuracy']:.4f} "
              f"elapsed={time.time() - t0:.1f}s")

    assert not np.isnan(oof_original).any()
    assert not np.isnan(oof_crossfit).any()

    original_summary = REPRO.summarize("Original (in-sample) two-stage arm C", oof_original, y_real)
    crossfit_summary = REPRO.summarize(f"Cross-fitted (inner KFold={inner_splits}) two-stage", oof_crossfit, y_real)

    reconciliation_delta = abs(original_summary["auroc"] - CITED_REFERENCE["auroc"])
    reconciles = reconciliation_delta <= RECONCILIATION_TOLERANCE
    print(f"\nRegression guard: original-arm AUROC {original_summary['auroc']:.6f} vs cited "
          f"{CITED_REFERENCE['auroc']:.6f} -> delta {reconciliation_delta:.6f} "
          f"({'RECONCILES' if reconciles else 'DOES NOT RECONCILE'} within tolerance {RECONCILIATION_TOLERANCE})")
    assert reconciles, (
        f"Original-arm reconstruction does not reconcile with the cited reference "
        f"(delta {reconciliation_delta:.6f} > tolerance {RECONCILIATION_TOLERANCE}) — "
        f"fold reproduction is not trustworthy, cross-fit comparison invalid until fixed."
    )

    auroc_lo_o, auroc_hi_o, _ = bootstrap_ci(y_real, oof_original, roc_auc_score)
    auprc_lo_o, auprc_hi_o, _ = bootstrap_ci(y_real, oof_original, average_precision_score)
    auroc_lo_c, auroc_hi_c, _ = bootstrap_ci(y_real, oof_crossfit, roc_auc_score)
    auprc_lo_c, auprc_hi_c, _ = bootstrap_ci(y_real, oof_crossfit, average_precision_score)

    paired_auroc = paired_bootstrap_auroc_delta(y_real, oof_crossfit, oof_original)
    # Reuse the same helper for AUPRC by temporarily swapping the scorer via
    # a thin local wrapper (helper is hardcoded to roc_auc_score internally,
    # so AUPRC delta is computed here directly rather than re-importing a
    # second near-identical helper).
    rng = np.random.default_rng(REPRO.RANDOM_STATE)
    auprc_deltas = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        yt = y_real[idx]
        if len(np.unique(yt)) < 2:
            continue
        auprc_deltas.append(average_precision_score(yt, oof_crossfit[idx]) - average_precision_score(yt, oof_original[idx]))
    auprc_delta_lo, auprc_delta_hi = np.percentile(auprc_deltas, [2.5, 97.5]) if auprc_deltas else (float("nan"), float("nan"))

    verdict_materially_different = bool(paired_auroc["ci_lo"] > 0 or paired_auroc["ci_hi"] < 0)

    results = {
        "meta": {
            "script": "scripts/crossfit_psyche_d_two_stage_pipeline.py",
            "debug_session": ".planning/debug/psyche-d-baseline-freeze.md",
            "inner_splits": inner_splits,
            "n_boot": N_BOOTSTRAP,
            "runtime_sec": time.time() - t0,
            "versions": {"pandas": pd.__version__, "numpy": np.__version__, "lightgbm": lgb.__version__, "sklearn": sklearn.__version__},
        },
        "cited_reference": CITED_REFERENCE,
        "regression_guard": {"reconciles": reconciles, "delta": reconciliation_delta, "tolerance": RECONCILIATION_TOLERANCE},
        "phase1_fold_metrics_full": phase1_fold_metrics_full,
        "phase1_fold_metrics_inner": phase1_fold_metrics_inner,
        "phase1_inner_mean_accuracy": float(np.mean([f["accuracy"] for f in phase1_fold_metrics_inner])),
        "phase1_full_mean_accuracy": float(np.mean([f["accuracy"] for f in phase1_fold_metrics_full])),
        "fold_coverage": fold_coverage,
        "original_arm": {
            **original_summary,
            "auroc_ci": [auroc_lo_o, auroc_hi_o],
            "auprc_ci": [auprc_lo_o, auprc_hi_o],
        },
        "crossfit_arm": {
            **crossfit_summary,
            "auroc_ci": [auroc_lo_c, auroc_hi_c],
            "auprc_ci": [auprc_lo_c, auprc_hi_c],
        },
        "paired_auroc_delta_crossfit_minus_original": paired_auroc,
        "paired_auprc_delta_crossfit_minus_original": {
            "mean_delta": float(np.mean(auprc_deltas)) if auprc_deltas else float("nan"),
            "ci_lo": float(auprc_delta_lo), "ci_hi": float(auprc_delta_hi), "n_draws": len(auprc_deltas),
        },
        "verdict_materially_different": verdict_materially_different,
    }

    print(f"\nParticipant coverage check: {sum(f['n_train_participants'] for f in fold_coverage)} total "
          f"outer-train-participant-slots across {REPRO.N_SPLITS} folds fully cross-fitted "
          f"(inner_splits={inner_splits}).")
    print(f"Paired AUROC delta (crossfit - original): {paired_auroc['mean_delta']:+.4f} "
          f"95% CI [{paired_auroc['ci_lo']:+.4f}, {paired_auroc['ci_hi']:+.4f}] n={paired_auroc['n_draws']} "
          f"-> {'MATERIALLY DIFFERENT (CI excludes zero)' if verdict_materially_different else 'NOT materially different (CI straddles zero)'}")

    return results


def write_report(results: dict) -> None:
    o = results["original_arm"]
    c = results["crossfit_arm"]
    pd_ = results["paired_auroc_delta_crossfit_minus_original"]
    pp = results["paired_auprc_delta_crossfit_minus_original"]
    verdict = "MATERIALLY DIFFERENT" if results["verdict_materially_different"] else "NOT materially different"
    lines = [
        "# PSYCHE-D Investigation #1 — Participant-Level Cross-Fitted Two-Stage Pipeline",
        "",
        f"Debug session: `{results['meta']['debug_session']}`",
        f"Script: `{results['meta']['script']}` (inner_splits={results['meta']['inner_splits']}, n_boot={results['meta']['n_boot']}, runtime={results['meta']['runtime_sec']:.1f}s)",
        "",
        "## The gap",
        "",
        "`audit_psyche_d_baseline_reproduction.py`'s two-stage arm (arm C, AUROC 0.6530) generates",
        "Phase-2 TRAINING-row intermediate-PHQ features using a Phase-1 model fit on that SAME outer",
        "fold's TRAIN participants — including, for any training participant with 2+ labeled quarters,",
        "that participant's own other quarter-end label(s). This is in-sample generation for the",
        "training side (confirmed by direct code trace). It was previously verified leakage-safe only",
        "for the EVAL side.",
        "",
        "## The fix",
        "",
        f"Participant-level inner cross-fitting (KFold, {results['meta']['inner_splits']} inner folds) applied",
        "ONLY to outer-TRAIN participants. Each inner fold's held-out participants get their intermediate",
        "features generated by a Phase-1 model (own imputer + own LightGBM-DART fit) that excludes them",
        "entirely. Outer-TEST participants are unchanged (already correct, test-time-equivalent).",
        "",
        "## Regression guard",
        "",
        f"Original-arm reconstruction AUROC = {o['auroc']:.6f} vs cited reference {results['cited_reference']['auroc']:.6f} "
        f"(delta {results['regression_guard']['delta']:.6f}, tolerance {results['regression_guard']['tolerance']}) "
        f"-> **{'RECONCILES' if results['regression_guard']['reconciles'] else 'DOES NOT RECONCILE'}**.",
        "",
        "## Results (pooled 5-fold OOF, n=10866)",
        "",
        "| Arm | AUROC | AUROC 95% CI | AUPRC | AUPRC 95% CI | Accuracy | Sensitivity | Specificity | TP/FP/TN/FN |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| Original (in-sample, arm C) | {o['auroc']:.4f} | [{o['auroc_ci'][0]:.4f}, {o['auroc_ci'][1]:.4f}] | {o['auprc']:.4f} | [{o['auprc_ci'][0]:.4f}, {o['auprc_ci'][1]:.4f}] | {o['accuracy']:.4f} | {o['sensitivity']:.4f} | {o['specificity']:.4f} | {o['tp']}/{o['fp']}/{o['tn']}/{o['fn']} |",
        f"| Cross-fitted | {c['auroc']:.4f} | [{c['auroc_ci'][0]:.4f}, {c['auroc_ci'][1]:.4f}] | {c['auprc']:.4f} | [{c['auprc_ci'][0]:.4f}, {c['auprc_ci'][1]:.4f}] | {c['accuracy']:.4f} | {c['sensitivity']:.4f} | {c['specificity']:.4f} | {c['tp']}/{c['fp']}/{c['tn']}/{c['fn']} |",
        "",
        f"Paired bootstrap AUROC delta (cross-fitted minus original): {pd_['mean_delta']:+.4f}, "
        f"95% CI [{pd_['ci_lo']:+.4f}, {pd_['ci_hi']:+.4f}] (n={pd_['n_draws']} valid draws).",
        f"Paired bootstrap AUPRC delta (cross-fitted minus original): {pp['mean_delta']:+.4f}, "
        f"95% CI [{pp['ci_lo']:+.4f}, {pp['ci_hi']:+.4f}] (n={pp['n_draws']} valid draws).",
        "",
        f"**Verdict: {verdict}** (paired AUROC 95% CI {'excludes' if results['verdict_materially_different'] else 'straddles'} zero).",
        "",
        f"Phase 1 mean accuracy: full-train (test-side/original-train-side) model = {results['phase1_full_mean_accuracy']:.4f}, "
        f"inner cross-fit models (train-side, cross-fitted) = {results['phase1_inner_mean_accuracy']:.4f}.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote {REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-splits", type=int, default=INNER_SPLITS)
    args = parser.parse_args()

    print(f"pandas={pd.__version__} numpy={np.__version__} lightgbm={lgb.__version__} sklearn={sklearn.__version__}")
    results = run(inner_splits=args.inner_splits)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {RESULTS_PATH}")
    write_report(results)


if __name__ == "__main__":
    main()
