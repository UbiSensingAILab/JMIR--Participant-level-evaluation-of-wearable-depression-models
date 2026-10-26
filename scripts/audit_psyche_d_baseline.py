#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), derived from the DiSCover Project
# (NCT03421223), CC BY-NC 4.0 — a separate institution's protocol,
# independent of this project's own ethics protocol (REB25-1587_MOD1).
# `research_datasets/` is gitignored; the parquet is local research data
# and is never committed to this repo.
"""PSYCHE-D baseline audit — PART A (forensic hypothesis-elimination sweep).

Companion to `train_psyche_d_severity_change.py`, which is the pipeline
under audit. This script performs an INDEPENDENT re-derivation of every
dataset/participant/target/CV/prediction fact the training script asserts
about itself, plus diagnostics the training script does not compute
(wearable-vs-non-wearable ablation, feature-matrix sanity, prediction-score
distributions, label-shuffle permutation test, grouped-vs-random-split
comparison, and a direct LightGBM-DART diagnostic arm). It does not import
or call the training script — every number here is derived fresh from the
raw parquet, so the two are cross-checks of each other, not one script
trusting the other's assertions.

See `.planning/debug/audit-psyche-d-baseline.md` (Part A, A1-A12) for the
full specification this script fulfils, and
`audit_psyche_d_baseline_report.md` for the narrative report built from
this script's + `audit_psyche_d_baseline_reproduction.py`'s output.

Requires `lightgbm` (not a pre-existing project dependency; installed for
this audit only — see report for the `libomp` macOS caveat). Every other
import is already a project dependency.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "research_datasets" / "old" / "psyche_d" / "anon_processed_df.parquet"
FE_PATH = Path(__file__).resolve().parent.parent / "research_datasets" / "old" / "psyche_d" / "feature_explanations.tsv"
OUT_PATH = Path(__file__).resolve().parent / "audit_psyche_d_baseline_results_part_a.json"
RANDOM_STATE = 42
N_SPLITS = 5

RESULTS: dict = {}


def section(name: str) -> None:
    print("\n" + "=" * 20 + f" {name} " + "=" * 20)


# ---------------------------------------------------------------------------
# Load + independently reconstruct the labeled subset (A1)
# ---------------------------------------------------------------------------
def load_and_audit_dataset() -> tuple[pd.DataFrame, list[str]]:
    section("A1 - dataset reconstruction")
    raw = pd.read_parquet(DATA_PATH)
    print("raw shape:", raw.shape)
    RESULTS["A1_raw_shape"] = list(raw.shape)
    assert raw.shape == (35694, 154), f"expected (35694, 154), got {raw.shape}"

    labeled = raw[raw["phq9_score_start"].notna() & raw["phq9_score_end"].notna()].copy()
    assert len(labeled) == 10866, f"expected 10866 labeled rows, got {len(labeled)}"
    RESULTS["A1_labeled_rows"] = len(labeled)

    # Cross-check: filtering on the *_cat_* columns instead of *_score_*
    # gives the identical row set (no silent NaN-pattern mismatch between
    # the two PHQ-9 representations).
    labeled_by_cat = raw[raw["phq9_cat_start"].notna() & raw["phq9_cat_end"].notna()]
    identical = set(labeled.index) == set(labeled_by_cat.index)
    print("score-filter vs cat-filter identical row set:", identical)
    RESULTS["A1_score_vs_cat_filter_identical"] = bool(identical)
    assert identical

    print("duplicate full rows:", int(labeled.duplicated(keep=False).sum()))
    print("duplicate index values:", int(labeled.index.duplicated(keep=False).sum()))
    RESULTS["A1_duplicate_rows"] = int(labeled.duplicated(keep=False).sum())
    RESULTS["A1_duplicate_index"] = int(labeled.index.duplicated(keep=False).sum())

    leakage_cols = ["phq9_score_start", "phq9_score_end", "phq9_cat_start", "phq9_cat_end"]
    feature_cols = [c for c in labeled.columns if c not in leakage_cols]
    assert len(feature_cols) == 150, f"expected 150 feature cols, got {len(feature_cols)}"
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(labeled[c])]
    assert not non_numeric, f"non-numeric feature columns found: {non_numeric}"
    RESULTS["A1_feature_col_count"] = len(feature_cols)

    nan_frac = labeled[feature_cols].isna().mean().sort_values(ascending=False)
    print("worst-missingness column:", nan_frac.index[0], f"{nan_frac.iloc[0]:.4f}")
    RESULTS["A1_worst_missingness_col"] = nan_frac.index[0]
    RESULTS["A1_worst_missingness_frac"] = float(nan_frac.iloc[0])
    RESULTS["A1_missingness_full"] = {k: float(v) for k, v in nan_frac.items()}
    assert abs(nan_frac.iloc[0] - 0.8666) < 0.001

    nunique = labeled[feature_cols].nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    print("constant feature cols:", constant_cols)
    RESULTS["A1_constant_cols"] = constant_cols
    assert not constant_cols, "found a constant feature column - would need investigation"

    near_constant = [
        (c, float(labeled[c].value_counts(normalize=True, dropna=True).iloc[0]))
        for c in feature_cols
        if len(labeled[c].value_counts(normalize=True, dropna=True)) > 0
        and labeled[c].value_counts(normalize=True, dropna=True).iloc[0] > 0.99
    ]
    print("near-constant cols (top value >99%):", near_constant)
    RESULTS["A1_near_constant_cols"] = near_constant

    inf_cols = {
        c: int(np.isinf(labeled[c].to_numpy(dtype=np.float64, na_value=np.nan)).sum())
        for c in feature_cols
        if np.isinf(labeled[c].to_numpy(dtype=np.float64, na_value=np.nan)).sum() > 0
    }
    print("columns with infinities:", inf_cols)
    RESULTS["A1_inf_cols"] = inf_cols
    assert not inf_cols

    return labeled, feature_cols


# ---------------------------------------------------------------------------
# A2 - participant-ID extraction audit
# ---------------------------------------------------------------------------
def audit_participant_ids(labeled: pd.DataFrame) -> np.ndarray:
    section("A2 - participant ID extraction audit")
    idx = labeled.index.to_series()
    underscore_counts = idx.str.count("_")
    print("underscore-count distribution in labeled index strings:")
    print(underscore_counts.value_counts())
    RESULTS["A2_underscore_count_distribution"] = {str(k): int(v) for k, v in underscore_counts.value_counts().items()}
    assert (underscore_counts == 1).all(), "found an index with != 1 underscore - split('_')[0] would be ambiguous"

    participant = idx.str.split("_").str[0]
    suffix = idx.str.split("_").str[1]
    n_unique = len(set(participant))
    assert n_unique == 4036, f"expected 4036 unique participants, got {n_unique}"
    print("unique participants:", n_unique)
    RESULTS["A2_n_unique_participants"] = n_unique

    participant_numeric_fail = pd.to_numeric(participant, errors="coerce").isna().sum()
    print("participant IDs failing numeric parse:", int(participant_numeric_fail))
    RESULTS["A2_participant_non_numeric"] = int(participant_numeric_fail)
    assert participant_numeric_fail == 0, "a participant ID is not purely numeric - grouping variable may not be a clean person ID"

    suffix_numeric = pd.to_numeric(suffix, errors="coerce")
    print("month-suffix value counts (labeled subset):")
    print(suffix_numeric.value_counts().sort_index())
    RESULTS["A2_suffix_value_counts_labeled"] = {str(k): int(v) for k, v in suffix_numeric.value_counts().sort_index().items()}
    assert set(suffix_numeric.unique()) == {3, 6, 9, 12}, "expected labeled rows only at quarterly checkpoints {3,6,9,12}"

    combo = participant.astype(str) + "_" + suffix.astype(str)
    n_dup_combo = int(combo.duplicated().sum())
    print("duplicate participant_month combos:", n_dup_combo)
    RESULTS["A2_duplicate_participant_month_combos"] = n_dup_combo
    assert n_dup_combo == 0

    samples_per_participant = pd.Series(participant.to_numpy()).value_counts()
    print("samples-per-participant: min/median/max:", samples_per_participant.min(), samples_per_participant.median(), samples_per_participant.max())
    RESULTS["A2_samples_per_participant"] = {
        "min": int(samples_per_participant.min()), "median": float(samples_per_participant.median()),
        "max": int(samples_per_participant.max()),
        "distribution": {str(k): int(v) for k, v in samples_per_participant.value_counts().sort_index().items()},
    }
    return participant.to_numpy()


# ---------------------------------------------------------------------------
# A3 - target reconstruction + ordinality
# ---------------------------------------------------------------------------
def audit_target(labeled: pd.DataFrame) -> np.ndarray:
    section("A3 - target reconstruction + ordinality check")
    delta_cat = labeled["phq9_cat_end"] - labeled["phq9_cat_start"]
    y = (delta_cat > 0).astype(int).to_numpy()
    n_pos = int(y.sum())
    assert n_pos == 2252, f"expected 2252 positives, got {n_pos}"
    print(f"positives: {n_pos}/{len(y)} = {n_pos/len(y):.4f}")
    RESULTS["A3_n_pos"] = n_pos
    RESULTS["A3_positive_rate"] = n_pos / len(y)

    trans = pd.crosstab(labeled["phq9_cat_start"], labeled["phq9_cat_end"])
    print("transition matrix (rows=cat_start, cols=cat_end):")
    print(trans)
    RESULTS["A3_transition_matrix"] = {str(k): {str(k2): int(v2) for k2, v2 in v.items()} for k, v in trans.to_dict().items()}
    RESULTS["A3_improved"] = int((delta_cat < 0).sum())
    RESULTS["A3_same"] = int((delta_cat == 0).sum())
    RESULTS["A3_worsened"] = int((delta_cat > 0).sum())

    # CRITICAL (A3): is the categorical encoding genuinely ordinal in the
    # direction the target formula assumes? Verify against the real score
    # ranges per category rather than assuming.
    score_by_cat = labeled.groupby("phq9_cat_start")["phq9_score_start"].agg(["min", "max", "mean", "count"])
    print("phq9_score_start range per phq9_cat_start bucket:")
    print(score_by_cat)
    RESULTS["A3_score_by_cat_start"] = score_by_cat.reset_index().to_dict(orient="records")
    mins, maxs = score_by_cat["min"].to_numpy(), score_by_cat["max"].to_numpy()
    monotonic_nonoverlap = all(maxs[i] < mins[i + 1] for i in range(len(mins) - 1))
    print("cat ranges strictly increasing & non-overlapping (standard PHQ-9 severity banding):", monotonic_nonoverlap)
    RESULTS["A3_ordinal_direction_confirmed"] = bool(monotonic_nonoverlap)
    assert monotonic_nonoverlap, "phq9_cat is NOT confirmed ordinal in the assumed direction - target formula would be suspect"

    return y


# ---------------------------------------------------------------------------
# CV helper shared by A6/A8/A9/A10/A11
# ---------------------------------------------------------------------------
def run_cv_auroc(X_sub: np.ndarray, y: np.ndarray, groups: np.ndarray, estimator_factory, stratified_group: bool = True, seed: int = RANDOM_STATE) -> dict:
    if stratified_group:
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        split_iter = splitter.split(X_sub, y, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        split_iter = splitter.split(X_sub, y)
    n = len(y)
    oof = np.full(n, np.nan)
    fold_info = []
    for fold_idx, (tr, te) in enumerate(split_iter):
        if stratified_group:
            overlap = set(groups[tr]) & set(groups[te])
            assert not overlap, f"participant leakage in fold {fold_idx}: {sorted(overlap)[:5]}"
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("clf", estimator_factory()),
        ])
        pipe.fit(X_sub[tr], y[tr])
        oof[te] = pipe.predict_proba(X_sub[te])[:, 1]
        fold_info.append({
            "fold": fold_idx + 1, "n_train": int(len(tr)), "n_test": int(len(te)),
            "n_train_participants": int(len(set(groups[tr]))) if stratified_group else None,
            "n_test_participants": int(len(set(groups[te]))) if stratified_group else None,
            "train_pos": int(y[tr].sum()), "test_pos": int(y[te].sum()), "test_pos_rate": float(y[te].mean()),
        })
    assert not np.isnan(oof).any()
    return {"auroc": float(roc_auc_score(y, oof)), "auprc": float(average_precision_score(y, oof)), "oof": oof, "fold_info": fold_info}


def lr_factory():
    return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)


def rf_factory():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)


# ---------------------------------------------------------------------------
# A6 - wearable vs non-wearable diagnostic
# ---------------------------------------------------------------------------
def audit_wearable_vs_nonwearable(labeled: pd.DataFrame, feature_cols: list[str], y: np.ndarray, participant_ids: np.ndarray) -> None:
    section("A6 - wearable vs non-wearable diagnostic (LogisticRegression)")
    fe = pd.read_csv(FE_PATH, sep="\t")
    wearable_cols = [c for c in fe.loc[fe["Category"].isin(["Steps", "Sleep"]), "Feature name"] if c in feature_cols]
    nonwearable_cols = [c for c in fe.loc[fe["Category"] == "Demographic", "Feature name"] if c in feature_cols]
    assert len(wearable_cols) + len(nonwearable_cols) == 150

    X_wearable = labeled[wearable_cols].to_numpy(dtype=np.float64)
    X_nonwearable = labeled[nonwearable_cols].to_numpy(dtype=np.float64)
    X_combined = labeled[wearable_cols + nonwearable_cols].to_numpy(dtype=np.float64)
    X_full = labeled[feature_cols].to_numpy(dtype=np.float64)

    for name, X_sub in [("wearable_only", X_wearable), ("nonwearable_only", X_nonwearable), ("combined", X_combined), ("full_150_reference", X_full)]:
        res = run_cv_auroc(X_sub, y, participant_ids, lr_factory)
        print(f"{name:25s} n_feat={X_sub.shape[1]:3d} AUROC={res['auroc']:.4f} AUPRC={res['auprc']:.4f}")
        RESULTS[f"A6_{name}"] = {"n_features": X_sub.shape[1], "auroc": res["auroc"], "auprc": res["auprc"]}


# ---------------------------------------------------------------------------
# A7 - feature matrix sanity
# ---------------------------------------------------------------------------
def audit_feature_matrix(labeled: pd.DataFrame, feature_cols: list[str]) -> None:
    section("A7 - feature matrix sanity")
    Xdf = labeled[feature_cols]
    dup_rows = int(Xdf.duplicated(keep=False).sum())
    print("duplicate X rows (all 150 features identical):", dup_rows)
    RESULTS["A7_duplicate_X_rows"] = dup_rows

    corr = Xdf.corr(numeric_only=True)
    np.fill_diagonal(corr.values, 0)
    perfect_pairs = [
        (c1, c2, float(corr.loc[c1, c2]))
        for i, c1 in enumerate(feature_cols) for c2 in feature_cols[i + 1:]
        if pd.notna(corr.loc[c1, c2]) and abs(corr.loc[c1, c2]) > 0.999
    ]
    print("feature pairs |corr| > 0.999:", perfect_pairs)
    RESULTS["A7_near_perfect_corr_pairs"] = perfect_pairs
    n_unique_rows = len(Xdf.drop_duplicates())
    print(f"unique feature-vector rows: {n_unique_rows}/{len(Xdf)}")
    RESULTS["A7_n_unique_feature_rows"] = n_unique_rows


# ---------------------------------------------------------------------------
# A9 - prediction distributions (+ A8 fold breakdown reused from A6's full-150 run)
# ---------------------------------------------------------------------------
def audit_prediction_distributions(labeled: pd.DataFrame, feature_cols: list[str], y: np.ndarray, participant_ids: np.ndarray) -> dict:
    section("A8/A9 - CV fold breakdown + prediction-distribution audit")
    X = labeled[feature_cols].to_numpy(dtype=np.float64)
    res_lr = run_cv_auroc(X, y, participant_ids, lr_factory)
    for f in res_lr["fold_info"]:
        print(f)
    RESULTS["A8_fold_info"] = res_lr["fold_info"]

    res_rf = run_cv_auroc(X, y, participant_ids, rf_factory)
    for name, res in [("logistic_regression", res_lr), ("random_forest", res_rf)]:
        proba = res["oof"]
        proba_pos, proba_neg = proba[y == 1], proba[y == 0]
        stats = {
            "min": float(proba.min()), "max": float(proba.max()), "mean": float(proba.mean()), "std": float(proba.std()),
            "n_unique": int(len(np.unique(proba))),
            "frac_ge_half": float((proba >= 0.5).mean()),
            "mean_score_given_y1": float(proba_pos.mean()), "mean_score_given_y0": float(proba_neg.mean()),
        }
        print(f"--- {name} --- {stats}")
        RESULTS[f"A9_{name}"] = stats
    return {"X": X, "res_lr": res_lr}


# ---------------------------------------------------------------------------
# A10 - label-shuffle permutation test (critical leakage check)
# ---------------------------------------------------------------------------
def audit_permutation_test(X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray) -> None:
    section("A10 - LABEL-SHUFFLE PERMUTATION TEST")
    aurocs = []
    for seed in range(5):
        rng = np.random.default_rng(1000 + seed)
        y_shuffled = rng.permutation(y)
        res = run_cv_auroc(X, y_shuffled, participant_ids, lr_factory)
        print(f"  permutation seed={seed}: AUROC={res['auroc']:.4f}")
        aurocs.append(res["auroc"])
    aurocs = np.array(aurocs)
    print(f"permutation AUROC mean={aurocs.mean():.4f} std={aurocs.std():.4f}")
    RESULTS["A10_permutation_aurocs"] = aurocs.tolist()
    RESULTS["A10_permutation_auroc_mean"] = float(aurocs.mean())
    RESULTS["A10_permutation_auroc_std"] = float(aurocs.std())


# ---------------------------------------------------------------------------
# A11 - random row-split vs grouped CV (debugging only)
# ---------------------------------------------------------------------------
def audit_random_split(X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray, grouped_auroc: float) -> None:
    section("A11 - random row-split vs grouped CV (debugging only)")
    res = run_cv_auroc(X, y, participant_ids, lr_factory, stratified_group=False)
    print(f"grouped CV AUROC={grouped_auroc:.4f}  random-split AUROC={res['auroc']:.4f}")
    RESULTS["A11_grouped_cv_auroc"] = grouped_auroc
    RESULTS["A11_random_split_auroc"] = res["auroc"]


# ---------------------------------------------------------------------------
# A12 - baseline PHQ severity conditional check
# ---------------------------------------------------------------------------
def audit_conditional_worsening(labeled: pd.DataFrame, y: np.ndarray, participant_ids: np.ndarray) -> None:
    section("A12 - baseline PHQ severity / sample-order / month conditional check")
    cat_start = labeled["phq9_cat_start"].to_numpy()
    by_cat = {}
    for cat in sorted(np.unique(cat_start)):
        mask = cat_start == cat
        rate = float(y[mask].mean())
        print(f"  phq9_cat_start={cat:.0f}: n={int(mask.sum()):5d} worsened_rate={rate:.4f}")
        by_cat[str(cat)] = {"n": int(mask.sum()), "rate": rate}
    RESULTS["A12_worsen_rate_by_cat_start"] = by_cat

    order = labeled.groupby(participant_ids).cumcount().to_numpy() + 1
    by_order = {}
    for k in sorted(np.unique(order)):
        mask = order == k
        rate = float(y[mask].mean())
        print(f"  sample_number_within_participant={k}: n={int(mask.sum()):5d} worsened_rate={rate:.4f}")
        by_order[str(k)] = {"n": int(mask.sum()), "rate": rate}
    RESULTS["A12_worsen_rate_by_sample_order"] = by_order

    month = pd.to_numeric(labeled.index.to_series().str.split("_").str[1]).to_numpy()
    by_month = {}
    for m in sorted(np.unique(month)):
        mask = month == m
        rate = float(y[mask].mean())
        print(f"  month_suffix={m}: n={int(mask.sum()):5d} worsened_rate={rate:.4f}")
        by_month[str(m)] = {"n": int(mask.sum()), "rate": rate}
    RESULTS["A12_worsen_rate_by_month"] = by_month


# ---------------------------------------------------------------------------
# B6 - diagnostic-only direct LightGBM-DART on the 150-feature matrix
# ---------------------------------------------------------------------------
def diagnostic_lgbm_dart(labeled: pd.DataFrame, feature_cols: list[str], y: np.ndarray, participant_ids: np.ndarray) -> None:
    section("B6 - diagnostic-only LightGBM-DART (NOT the official two-phase reproduction)")
    fe = pd.read_csv(FE_PATH, sep="\t")
    wearable_cols = [c for c in fe.loc[fe["Category"].isin(["Steps", "Sleep"]), "Feature name"] if c in feature_cols]
    nonwearable_cols = [c for c in fe.loc[fe["Category"] == "Demographic", "Feature name"] if c in feature_cols]
    feature_cols_with_baseline = feature_cols + ["phq9_score_start", "phq9_cat_start"]

    def lgbm_dart_factory():
        return lgb.LGBMClassifier(boosting_type="dart", n_estimators=300, random_state=RANDOM_STATE, is_unbalance=True, verbosity=-1, n_jobs=-1)

    def run_cv_lgbm(cols: list[str]) -> np.ndarray:
        skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        X = labeled[cols].to_numpy(dtype=np.float64)
        n = len(y)
        oof = np.full(n, np.nan)
        for tr, te in skf.split(X, y, groups=participant_ids):
            assert not (set(participant_ids[tr]) & set(participant_ids[te]))
            imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
            X_tr, X_te = imputer.fit_transform(X[tr]), imputer.transform(X[te])
            clf = lgbm_dart_factory()
            clf.fit(X_tr, y[tr])
            oof[te] = clf.predict_proba(X_te)[:, 1]
        assert not np.isnan(oof).any()
        return oof

    for name, cols in [
        ("lgbm_dart_150features", feature_cols),
        ("lgbm_dart_wearable_only", wearable_cols),
        ("lgbm_dart_nonwearable_only", nonwearable_cols),
        ("lgbm_dart_with_baseline_severity", feature_cols_with_baseline),
    ]:
        oof = run_cv_lgbm(cols)
        auroc, auprc = roc_auc_score(y, oof), average_precision_score(y, oof)
        predicted = (oof >= 0.5).astype(int)
        tp = int(((y == 1) & (predicted == 1)).sum())
        fp = int(((y == 0) & (predicted == 1)).sum())
        tn = int(((y == 0) & (predicted == 0)).sum())
        fn = int(((y == 1) & (predicted == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        acc = (tp + tn) / len(y)
        print(f"{name:35s} n_feat={len(cols):3d} AUROC={auroc:.4f} AUPRC={auprc:.4f} acc={acc:.4f} sens={sens:.4f} spec={spec:.4f}")
        RESULTS[f"B6_{name}"] = {
            "n_features": len(cols), "auroc": float(auroc), "auprc": float(auprc), "accuracy": float(acc),
            "sensitivity": float(sens), "specificity": float(spec), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }


def main() -> None:
    print(f"pandas={pd.__version__} numpy={np.__version__} lightgbm={lgb.__version__}")
    labeled, feature_cols = load_and_audit_dataset()
    participant_ids = audit_participant_ids(labeled)
    y = audit_target(labeled)
    audit_wearable_vs_nonwearable(labeled, feature_cols, y, participant_ids)
    audit_feature_matrix(labeled, feature_cols)
    ctx = audit_prediction_distributions(labeled, feature_cols, y, participant_ids)
    audit_permutation_test(ctx["X"], y, participant_ids)
    audit_random_split(ctx["X"], y, participant_ids, ctx["res_lr"]["auroc"])
    audit_conditional_worsening(labeled, y, participant_ids)
    diagnostic_lgbm_dart(labeled, feature_cols, y, participant_ids)

    OUT_PATH.write_text(json.dumps(RESULTS, indent=2, default=str))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
