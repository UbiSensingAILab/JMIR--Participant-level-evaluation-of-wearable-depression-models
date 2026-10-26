#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), derived from the DiSCover Project
# (NCT03421223), CC BY-NC 4.0 — a separate institution's protocol,
# independent of this project's own ethics protocol (REB25-1587_MOD1).
# `research_datasets/` is gitignored; the parquet is local research data
# and is never committed to this repo.
"""PSYCHE-D accuracy-vs-sensitivity threshold sweep.

DIAGNOSTIC ONLY. Answers one question honestly: what does it cost, in missed
worsening cases, to push the worsened-depression classifier's raw accuracy
up to 75% / 80% / 85% / 90%? No production pipeline, dashboard, frontend or
API code changes.

Reuses (never re-implements) the leakage-safe two-phase reproduction
pipeline from `audit_psyche_d_baseline_reproduction.py` (REPRO below):
`load_raw`, `build_phase1_input_cols`, `build_phase2_generated_features`,
`phase1_dart_factory`, `phase2_dart_factory`, and the constants `PROBA_COLS`,
`LEAKAGE_COLS`, `RANDOM_STATE`, `N_SPLITS` — same
`sys.path.insert(0, str(Path(__file__).resolve().parent))` cross-script
convention as `simulate_psyche_d_fl_soft_voting.py`.

Only the "Phase 2 WITH generated intermediate features" arm is reproduced
(the reproduction's main/best arm, cited full-data AUROC 0.6529684832943685)
across the identical `StratifiedGroupKFold(5)` protocol, grouped by
`user_id`, so folds and numbers are directly comparable. The "without
intermediate" ablation arm is intentionally skipped — not needed to answer
this question, and skipping it roughly halves runtime.

Unlike the reproduction script, this one pools the raw out-of-fold predicted
probability vector (not just a 0.5-threshold summary) so it can be swept
across thresholds — pure arithmetic on an already-computed array, no
retraining, so the sweep itself is near-instant once the OOF vector exists.

No participant ID, per-participant row, or per-row export is ever written to
the results JSON or the markdown report — aggregate metrics only, per this
dataset's pre-registered no-per-participant-export rule.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_psyche_d_baseline_reproduction as REPRO  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "threshold_sweep_psyche_d_severity_change_results.json"
REPORT_PATH = SCRIPT_DIR / "threshold_sweep_psyche_d_severity_change_report.md"

# Cited verbatim from scripts/audit_psyche_d_baseline_results.json, key
# part_b_reproduction.phase2_with_intermediate — the reconciliation target.
CITED_REFERENCE = {
    "auroc": 0.6529684832943685,
    "accuracy": 0.6175225473955457,
    "sensitivity": 0.5954706927175843,
    "specificity": 0.6232876712328768,
    "tp": 1341, "fp": 3245, "tn": 5369, "fn": 911,
    "source": "scripts/audit_psyche_d_baseline_results.json:part_b_reproduction.phase2_with_intermediate",
}
RECONCILIATION_TOLERANCE = 0.002
TARGET_ACCURACIES = [0.75, 0.80, 0.85, 0.90]


def confusion_at_threshold(y: np.ndarray, proba: np.ndarray, thr: float) -> dict:
    pred = (proba >= thr).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    n = len(y)
    acc = (tp + tn) / n
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) and not np.isnan(prec) and not np.isnan(sens) and (prec + sens) > 0 else float("nan")
    return {
        "threshold": round(float(thr), 4), "accuracy": acc, "sensitivity": sens,
        "specificity": spec, "precision": prec, "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def run_phase2_with_intermediate_oof() -> tuple[np.ndarray, np.ndarray]:
    """Reproduces only REPRO.run_reproduction()'s 'with intermediate' arm,
    but returns the raw pooled OOF probability vector + y instead of a
    0.5-threshold summary, so callers can sweep thresholds."""
    raw = REPRO.load_raw()
    phase1_input_cols = REPRO.build_phase1_input_cols(raw)
    phase2_base_cols = [c for c in raw.columns if c not in REPRO.LEAKAGE_COLS and c not in ("user_id", "user_mth", "ref_label")]
    assert len(phase2_base_cols) == 150

    labeled = raw[raw["ref_label"]].copy()
    participant_ids = labeled["user_id"].to_numpy()
    delta_cat = labeled["phq9_cat_end"] - labeled["phq9_cat_start"]
    y = (delta_cat > 0).astype(int).to_numpy()
    y_series = pd.Series(y, index=labeled.index)

    skf = StratifiedGroupKFold(n_splits=REPRO.N_SPLITS, shuffle=True, random_state=REPRO.RANDOM_STATE)
    n = len(y)
    oof = np.full(n, np.nan)

    for fold_idx, (tr_pos, te_pos) in enumerate(skf.split(np.zeros(n), y, groups=participant_ids)):
        train_idx = labeled.index[tr_pos]
        test_idx = labeled.index[te_pos]
        train_participants = set(participant_ids[tr_pos])
        test_participants = set(participant_ids[te_pos])
        assert not (train_participants & test_participants), f"participant leakage fold {fold_idx}"

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

        imputer2 = SimpleImputer(strategy="mean", keep_empty_features=True)
        X2_train = imputer2.fit_transform(X2_full.loc[tr_mask2])
        X2_test = imputer2.transform(X2_full.loc[te_mask2])
        sw2 = compute_sample_weight(class_weight="balanced", y=y2.loc[tr_mask2])
        clf2 = REPRO.phase2_dart_factory()
        clf2.fit(X2_train, y2.loc[tr_mask2], sample_weight=sw2)
        oof[te_pos] = clf2.predict_proba(X2_test)[:, 1]

        print(f"fold {fold_idx + 1}/{REPRO.N_SPLITS} done "
              f"(phase1_train_rows={int(phase1_train_mask.sum())}, "
              f"phase2_train_rows={int(tr_mask2.sum())}, phase2_test_rows={int(te_mask2.sum())})")

    assert not np.isnan(oof).any(), "OOF vector has unfilled positions"
    return oof, y


def main() -> None:
    t0 = time.time()
    print(f"pandas={pd.__version__} numpy={np.__version__}")

    oof, y = run_phase2_with_intermediate_oof()
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos

    auroc = float(roc_auc_score(y, oof))
    auprc = float(average_precision_score(y, oof))
    print(f"Pooled OOF: n={n} n_pos={n_pos} n_neg={n_neg} AUROC={auroc:.4f} AUPRC={auprc:.4f}")

    majority_baseline_accuracy = n_neg / n  # predict "not worsened" for every row

    default_at_05 = confusion_at_threshold(y, oof, 0.5)
    delta_auroc = auroc - CITED_REFERENCE["auroc"]
    delta_acc_05 = default_at_05["accuracy"] - CITED_REFERENCE["accuracy"]
    reconciled = abs(delta_auroc) <= RECONCILIATION_TOLERANCE
    print(f"0.5-threshold: accuracy={default_at_05['accuracy']:.4f} sens={default_at_05['sensitivity']:.4f} "
          f"spec={default_at_05['specificity']:.4f} | reconciliation vs cited AUROC 0.6530: "
          f"delta={delta_auroc:+.4f} ({'OK' if reconciled else 'MISMATCH'})")

    # Sweep thresholds every 0.01 from 0.0 to 1.0 inclusive (101 points).
    thresholds = np.round(np.arange(0.0, 1.0001, 0.01), 4)
    sweep = [confusion_at_threshold(y, oof, thr) for thr in thresholds]

    # For each target accuracy, find the lowest threshold that reaches it
    # (favors the least sensitivity-destructive way to hit the target, since
    # accuracy generally rises monotonically-ish as threshold -> 1 on this
    # imbalanced dataset). "Not reachable" is reported honestly if no swept
    # threshold clears the target without sensitivity collapsing to 0.
    target_rows = []
    for target in TARGET_ACCURACIES:
        reaching = [r for r in sweep if r["accuracy"] >= target]
        if not reaching:
            target_rows.append({"target_accuracy": target, "reachable": False, "note": "no threshold in [0,1] reaches this accuracy"})
            continue
        # Prefer the reaching threshold with the highest sensitivity (least damage).
        best = max(reaching, key=lambda r: (r["sensitivity"] if not np.isnan(r["sensitivity"]) else -1))
        target_rows.append({
            "target_accuracy": target, "reachable": True,
            "threshold": best["threshold"], "accuracy": best["accuracy"],
            "sensitivity": best["sensitivity"], "specificity": best["specificity"],
            "tp": best["tp"], "fp": best["fp"], "tn": best["tn"], "fn": best["fn"],
            "note": "0% sensitivity (majority-class classifier)" if (np.isnan(best["sensitivity"]) or best["sensitivity"] == 0) else "",
        })

    results = {
        "pandas_version": pd.__version__, "numpy_version": np.__version__,
        "n_rows": n, "n_positive": n_pos, "n_negative": n_neg,
        "auroc": auroc, "auprc": auprc,
        "majority_baseline_accuracy": majority_baseline_accuracy,
        "default_threshold_0.5": default_at_05,
        "cited_reference": CITED_REFERENCE,
        "reconciliation": {"delta_auroc": delta_auroc, "delta_accuracy_at_0.5": delta_acc_05, "within_tolerance": reconciled, "tolerance": RECONCILIATION_TOLERANCE},
        "target_accuracy_tradeoffs": target_rows,
        "full_sweep": sweep,
        "runtime_sec": round(time.time() - t0, 1),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {RESULTS_PATH}")

    lines = []
    lines.append("# PSYCHE-D Threshold Sweep — Accuracy vs. Sensitivity Trade-off\n")
    lines.append("PSYCHE-D dataset (Zenodo 5085146, DiSCover NCT03421223, CC BY-NC 4.0), separate institution's protocol, independent of this project's REB25-1587_MOD1. Diagnostic only — no production pipeline, dashboard, frontend or API code changed. Aggregate metrics only, no participant-level data.\n")
    lines.append(f"**Setup:** Phase2-with-intermediate arm only, `StratifiedGroupKFold(5)` grouped by `user_id`, identical protocol to `audit_psyche_d_baseline_reproduction.py`. n={n} labeled rows ({n_pos} positive / {n_neg} negative, {n_pos/n*100:.1f}% worsened).\n")
    lines.append(f"**Reconciliation:** measured AUROC {auroc:.4f} vs. cited full-data reference {CITED_REFERENCE['auroc']:.4f} (delta {delta_auroc:+.4f}, tolerance ±{RECONCILIATION_TOLERANCE}) — **{'reconciled' if reconciled else 'MISMATCH — investigate'}**.\n")
    lines.append(f"**Majority-class baseline** (predict \"not worsened\" for every row): accuracy = {majority_baseline_accuracy*100:.1f}%.\n")
    lines.append(f"**Default 0.5 threshold:** accuracy {default_at_05['accuracy']*100:.1f}%, sensitivity {default_at_05['sensitivity']*100:.1f}%, specificity {default_at_05['specificity']*100:.1f}% — below the majority-class baseline because training uses class-balanced sample weights optimizing sensitivity/specificity balance, not raw accuracy.\n")
    lines.append("## Accuracy targets vs. sensitivity cost\n")
    lines.append("| Target accuracy | Reachable? | Threshold | Actual accuracy | Sensitivity (recall on worsened cases) | Specificity | TP/FP/TN/FN | Note |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in target_rows:
        if row["reachable"]:
            lines.append(f"| {row['target_accuracy']*100:.0f}% | Yes | {row['threshold']:.2f} | {row['accuracy']*100:.1f}% | {row['sensitivity']*100:.1f}% | {row['specificity']*100:.1f}% | {row['tp']}/{row['fp']}/{row['tn']}/{row['fn']} | {row['note']} |")
        else:
            lines.append(f"| {row['target_accuracy']*100:.0f}% | **No** | — | — | — | — | — | {row['note']} |")
    lines.append("")
    lines.append("## Honest read\n")
    reachable_summary = ", ".join(
        f"{r['target_accuracy']*100:.0f}% costs sensitivity down to {r['sensitivity']*100:.1f}%"
        for r in target_rows if r["reachable"]
    )
    lines.append(f"Pushing the decision threshold up trades sensitivity for accuracy on this ~{n_pos/n*100:.0f}%-positive dataset: {reachable_summary if reachable_summary else 'no target accuracy above the default was reachable at any threshold'}. "
                 f"A model that hits 90% accuracy here by predicting almost everyone as \"not worsened\" is not clinically useful for a screening tool — it would miss the majority (or all) of the mothers whose depression actually worsened. "
                 f"AUROC ({auroc:.4f}) is threshold-independent and is the number that reflects real discriminative signal; accuracy at any single threshold is a downstream reporting choice, not a measure of model quality on an imbalanced clinical task like this one.\n")
    lines.append(f"\nRuntime: {results['runtime_sec']}s.")
    REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
