#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), derived from the DiSCover Project
# (NCT03421223), CC BY-NC 4.0 — a separate institution's protocol,
# independent of this project's own ethics protocol (REB25-1587_MOD1).
# `research_datasets/` is gitignored; the parquet is local research data
# and is never committed to this repo.
"""PSYCHE-D controlled feature-source ablation.

Investigation #3 of the `psyche-d-baseline-freeze` debug session
(`.planning/debug/psyche-d-baseline-freeze.md`). NOT an optimization
exercise — the goal is to determine whether the ~0.69-0.71 signal comes
primarily from baseline PHQ severity, the PSYCHE-D intermediate-prediction
mechanism, or wearable/context information, using six feature-source arms
on the SAME folds.

Built on the NOW-VALIDATED cross-fitted intermediate generation from
Investigation #1 (`crossfit_psyche_d_two_stage_pipeline.py`), not the
original in-sample arm C — Investigation #1 found the in-sample gap didn't
materially change AUROC, but the cross-fitted generation is the
methodologically correct one going forward, so it is what this ablation's
intermediate-prediction arms are built from.

SIX ARMS (all on the identical `StratifiedGroupKFold(5)` fold split as
Investigations #1/#2, pooled OOF, `REPRO.phase2_dart_factory()` for every
arm so any AUROC difference is attributable to feature source, not model
config):

  1. wearable_context_only        — 150 base features (Sleep+Steps+Demographic/LMC).
  2. baseline_phq_only             — phq9_score_start + phq9_cat_start (2 cols).
  3. wearable_context_plus_baseline_phq — 1 + 2 (152 cols, = arm B).
  4. intermediate_prediction_only  — cat_m1, cat_m2, proba_cat_{0-4}_m1/_m2
     (12 cols). `cat_m0` is DELIBERATELY EXCLUDED from this arm: per
     `REPRO.build_phase2_generated_features`, `cat_m0` is a direct copy of
     the row's own real `phq9_cat_start` value, NOT a Phase-1-generated
     prediction — including it here would silently duplicate arm 2's
     signal inside what's meant to be a "generated-prediction-only" arm.
  5. baseline_phq_plus_intermediate — 2 + 4 (14 cols).
  6. full_validated_feature_set    — 1 + 2 + 4 (164 cols). This is the
     "arm D" freeze candidate referenced in the debug session's grounding
     section.

REUSE, NOT REIMPLEMENTATION. Imports (never copies) `load_raw`,
`LEAKAGE_COLS`, `RANDOM_STATE`, `N_SPLITS`, `PROBA_COLS`, `PROBA_COLS_M1`,
`PROBA_COLS_M2`, `build_phase2_generated_features`, `phase2_dart_factory`,
`summarize` from `audit_psyche_d_baseline_reproduction` (`REPRO`);
`fit_phase1`, `generate_intermediate`, `INNER_SPLITS` from
`crossfit_psyche_d_two_stage_pipeline` (`CROSSFIT` — that script is a new
script from this same debug session, not modified here, only imported);
`bootstrap_ci`, `N_BOOTSTRAP` from `train_psyche_d_severity_change`. The
per-fold cross-fit generation loop below structurally mirrors CROSSFIT's own
loop (same two-stage design, same fold protocol) because it needs the SAME
generated columns to build SIX Phase-2 matrices per fold instead of one —
this is the same "own fold loop calling shared REPRO/CROSSFIT helpers"
pattern already used by `threshold_sweep_psyche_d_severity_change.py` and
`tune_psyche_d_severity_change_models.py`, not a copy of CROSSFIT's loop
body itself (no LightGBM/imputer fitting code is duplicated — it's all
delegated to `fit_phase1`/`generate_intermediate`).

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
from sklearn.model_selection import KFold, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_psyche_d_baseline_reproduction as REPRO  # noqa: E402
import crossfit_psyche_d_two_stage_pipeline as CROSSFIT  # noqa: E402
from train_psyche_d_severity_change import bootstrap_ci, N_BOOTSTRAP  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "ablation_psyche_d_severity_change_feature_sources_results.json"
REPORT_PATH = SCRIPT_DIR / "ablation_psyche_d_severity_change_feature_sources_report.md"

BASELINE_PHQ_COLS = ["phq9_score_start", "phq9_cat_start"]
INTERMEDIATE_ONLY_COLS = ["cat_m1", "cat_m2"] + REPRO.PROBA_COLS_M1 + REPRO.PROBA_COLS_M2
assert len(INTERMEDIATE_ONLY_COLS) == 12
assert "cat_m0" not in INTERMEDIATE_ONLY_COLS


def fit_eval_arm(X_full: pd.DataFrame, cols: list[str], y: pd.Series, tr_mask, te_mask, sw) -> np.ndarray:
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    X_tr = imputer.fit_transform(X_full.loc[tr_mask, cols])
    X_te = imputer.transform(X_full.loc[te_mask, cols])
    clf = REPRO.phase2_dart_factory()
    clf.fit(X_tr, y.loc[tr_mask], sample_weight=sw)
    return clf.predict_proba(X_te)[:, 1]


def run(inner_splits: int = CROSSFIT.INNER_SPLITS) -> dict:
    t0 = time.time()
    raw = REPRO.load_raw()
    phase1_input_cols = REPRO.build_phase1_input_cols(raw)
    phase2_base_cols = [c for c in raw.columns if c not in REPRO.LEAKAGE_COLS and c not in ("user_id", "user_mth", "ref_label")]
    assert len(phase2_base_cols) == 150

    arm_cols = {
        "wearable_context_only": phase2_base_cols,
        "baseline_phq_only": BASELINE_PHQ_COLS,
        "wearable_context_plus_baseline_phq": phase2_base_cols + BASELINE_PHQ_COLS,
        "intermediate_prediction_only": INTERMEDIATE_ONLY_COLS,
        "baseline_phq_plus_intermediate": BASELINE_PHQ_COLS + INTERMEDIATE_ONLY_COLS,
        "full_validated_feature_set": phase2_base_cols + BASELINE_PHQ_COLS + INTERMEDIATE_ONLY_COLS,
    }
    for name, cols in arm_cols.items():
        assert len(cols) == len(set(cols)), f"{name}: duplicate columns"
    print("Arm feature counts:", {k: len(v) for k, v in arm_cols.items()})

    labeled = raw[raw["ref_label"]].copy()
    participant_ids = labeled["user_id"].to_numpy()
    delta_cat = labeled["phq9_cat_end"] - labeled["phq9_cat_start"]
    y_real = (delta_cat > 0).astype(int).to_numpy()
    y_series = pd.Series(y_real, index=labeled.index)
    n = len(y_real)

    skf = StratifiedGroupKFold(n_splits=REPRO.N_SPLITS, shuffle=True, random_state=REPRO.RANDOM_STATE)
    oof = {name: np.full(n, np.nan) for name in arm_cols}

    for fold_idx, (tr_pos, te_pos) in enumerate(skf.split(np.zeros(n), y_real, groups=participant_ids)):
        train_idx = labeled.index[tr_pos]
        test_idx = labeled.index[te_pos]
        train_participants = sorted(set(participant_ids[tr_pos]))
        test_participants = set(participant_ids[te_pos])
        assert not (set(train_participants) & test_participants), f"participant leakage fold {fold_idx}"

        # clf1_full: test-side generation (unchanged, correct) + regression-guard parity with CROSSFIT.
        phase1_train_mask_full = raw["ref_label"] & raw["user_id"].isin(train_participants)
        imputer1_full, clf1_full = CROSSFIT.fit_phase1(raw, phase1_input_cols, phase1_train_mask_full)

        unlabeled_mask = ~raw["ref_label"]
        test_participant_unlabeled_mask = unlabeled_mask & raw["user_id"].isin(test_participants)

        cat_end_crossfit = raw["phq9_cat_end"].copy()
        proba_crossfit = pd.DataFrame(0.0, index=raw.index, columns=REPRO.PROBA_COLS)
        CROSSFIT.generate_intermediate(raw, phase1_input_cols, imputer1_full, clf1_full, test_participant_unlabeled_mask, cat_end_crossfit, proba_crossfit)

        inner_kf = KFold(n_splits=inner_splits, shuffle=True, random_state=REPRO.RANDOM_STATE * 10_000 + fold_idx)
        train_participants_arr = np.array(train_participants)
        covered = np.zeros(len(train_participants_arr), dtype=bool)
        for inner_tr_pos, inner_ho_pos in inner_kf.split(train_participants_arr):
            inner_train_participants = set(train_participants_arr[inner_tr_pos])
            inner_holdout_participants = set(train_participants_arr[inner_ho_pos])
            covered[inner_ho_pos] = True
            inner_train_mask = raw["ref_label"] & raw["user_id"].isin(inner_train_participants)
            imputer1_inner, clf1_inner = CROSSFIT.fit_phase1(raw, phase1_input_cols, inner_train_mask)
            inner_holdout_unlabeled_mask = unlabeled_mask & raw["user_id"].isin(inner_holdout_participants)
            CROSSFIT.generate_intermediate(raw, phase1_input_cols, imputer1_inner, clf1_inner, inner_holdout_unlabeled_mask, cat_end_crossfit, proba_crossfit)
        assert covered.all(), f"fold {fold_idx}: inner KFold did not cover every outer-train participant"

        fold_labeled_idx = train_idx.append(test_idx)
        gen_feats = REPRO.build_phase2_generated_features(raw, fold_labeled_idx, cat_end_crossfit, proba_crossfit)

        # phase2_base_cols excludes LEAKAGE_COLS (which includes the real
        # phq9_score_start/phq9_cat_start columns) by construction in REPRO —
        # correct for the wearable_context_only arm, but the baseline-PHQ
        # arms below need those two real columns added back explicitly.
        X2_base = raw.loc[fold_labeled_idx, phase2_base_cols]
        X2_baseline_phq = raw.loc[fold_labeled_idx, BASELINE_PHQ_COLS]
        X2_full = pd.concat([X2_base, X2_baseline_phq, gen_feats], axis=1)
        y2 = y_series.loc[fold_labeled_idx]
        tr_mask2 = X2_full.index.isin(train_idx)
        te_mask2 = X2_full.index.isin(test_idx)
        sw2 = compute_sample_weight(class_weight="balanced", y=y2.loc[tr_mask2])

        for name, cols in arm_cols.items():
            oof[name][te_pos] = fit_eval_arm(X2_full, cols, y2, tr_mask2, te_mask2, sw2)

        print(f"fold {fold_idx + 1}/{REPRO.N_SPLITS} done, elapsed={time.time() - t0:.1f}s")

    results = {"meta": {
        "script": "scripts/ablation_psyche_d_severity_change_feature_sources.py",
        "debug_session": ".planning/debug/psyche-d-baseline-freeze.md",
        "inner_splits": inner_splits,
        "n_boot": N_BOOTSTRAP,
        "runtime_sec": time.time() - t0,
        "versions": {"pandas": pd.__version__, "numpy": np.__version__, "lightgbm": lgb.__version__, "sklearn": sklearn.__version__},
        "arm_feature_counts": {k: len(v) for k, v in arm_cols.items()},
    }, "arms": {}}

    for name in arm_cols:
        vec = oof[name]
        assert not np.isnan(vec).any()
        summary = REPRO.summarize(name, vec, y_real)
        lo, hi, _ = bootstrap_ci(y_real, vec, roc_auc_score)
        alo, ahi, _ = bootstrap_ci(y_real, vec, average_precision_score)
        summary["auroc_ci"] = [lo, hi]
        summary["auprc_ci"] = [alo, ahi]
        results["arms"][name] = summary

    return results, oof, y_real


def write_report(results: dict) -> None:
    lines = [
        "# PSYCHE-D Investigation #3 — Controlled Feature-Source Ablation",
        "",
        f"Debug session: `{results['meta']['debug_session']}`",
        f"Script: `{results['meta']['script']}` (inner_splits={results['meta']['inner_splits']}, n_boot={results['meta']['n_boot']}, runtime={results['meta']['runtime_sec']:.1f}s)",
        "",
        "All six arms share the identical StratifiedGroupKFold(5) split and REPRO.phase2_dart_factory() —",
        "any AUROC difference is attributable to feature source only. Intermediate-prediction arms use",
        "Investigation #1's cross-fitted (leakage-corrected) generation, not the original in-sample arm C.",
        "",
        "## Results (pooled 5-fold OOF, n=10866)",
        "",
        "| Arm | n_feat | AUROC | AUROC 95% CI | AUPRC | Accuracy | Sensitivity | Specificity |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, s in results["arms"].items():
        nf = results["meta"]["arm_feature_counts"][name]
        lines.append(f"| {name} | {nf} | {s['auroc']:.4f} | [{s['auroc_ci'][0]:.4f}, {s['auroc_ci'][1]:.4f}] | {s['auprc']:.4f} | {s['accuracy']:.4f} | {s['sensitivity']:.4f} | {s['specificity']:.4f} |")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote {REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-splits", type=int, default=CROSSFIT.INNER_SPLITS)
    args = parser.parse_args()
    print(f"pandas={pd.__version__} numpy={np.__version__} lightgbm={lgb.__version__} sklearn={sklearn.__version__}")
    results, oof, y_real = run(inner_splits=args.inner_splits)
    for name, s in results["arms"].items():
        print(f"{name:38s} AUROC={s['auroc']:.4f} CI={s['auroc_ci']}")
    out = {**results, "_oof_vectors_not_exported": "raw OOF arrays intentionally excluded from JSON to keep file size sane; summaries only"}
    RESULTS_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {RESULTS_PATH}")
    write_report(results)


if __name__ == "__main__":
    main()
