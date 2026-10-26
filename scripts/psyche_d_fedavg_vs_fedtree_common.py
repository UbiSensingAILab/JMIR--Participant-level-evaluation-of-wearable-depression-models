#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), derived from the DiSCover Project
# (NCT03421223), CC BY-NC 4.0 -- a separate institution's protocol,
# independent of this project's own ethics protocol (REB25-1587_MOD1).
# `research_datasets/` is gitignored; the parquet is local research data
# and is never committed to this repo.
"""Shared infrastructure for the psyche-d-fedavg-vs-fedtree debug session
(`.planning/debug/psyche-d-fedavg-vs-fedtree.md`). NEW module, self-contained
-- does NOT import from any file belonging to the separate, still-unconfirmed
`.planning/debug/psyche-d-fl-experiment-design.md` session (those scripts use
StratifiedGroupKFold(5) pooled-OOF, a materially different protocol from this
session's ONE-TIME fixed participant train/validation/test split).

Reuses ONLY `audit_psyche_d_baseline_reproduction` (REPRO) -- explicitly
whitelisted, shared, reusable infrastructure from the (separate, already
archived) `psyche-d-baseline-freeze` session: `load_raw()`, `LEAKAGE_COLS`,
`RANDOM_STATE`.

Provides:
  - `load_arm_b_frame()`               -- the frozen 152-feature arm-B matrix + binary target
  - `build_fixed_split()`              -- ONE-TIME participant-level train/val/test split (not K-fold)
  - `assign_clients()`                 -- whole-participant K-way client partition of the TRAIN pool only
  - `federated_sufficient_stats()`     -- mean/std computed via aggregated (count, sum, sumsq) --
                                          identical whether given 1 group (centralized) or K groups
                                          (federated); never touches validation/test rows; never
                                          shares individual records across "clients" in the federated case
  - `apply_impute_scale()`             -- apply a given (mean, std) to impute + standardize a frame
  - `compute_metrics()` / `bootstrap_ci_auroc()` -- self-contained metrics (row-level bootstrap)
  - `SPLIT_SEED`, `TRAIN_FRAC`, `VAL_FRAC`, `TEST_FRAC`, `N_CLIENTS`, `PARTITION_SEEDS`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_psyche_d_baseline_reproduction as REPRO  # noqa: E402

BASELINE_SEVERITY_COLS = ["phq9_score_start", "phq9_cat_start"]

# --- Fixed protocol constants (documented, not tuned) ---
SPLIT_SEED = 20260928       # date-derived, fixed once, never re-rolled after inspecting results
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
N_CLIENTS = 5                # cross-silo range already used elsewhere in this project's FL work
PARTITION_SEEDS = [1, 2, 3]  # "several seeds" per task Step 3 -- one K, three partition seeds
N_BOOTSTRAP = 2000

ARM_B_CENTRALIZED_LGBM_DART_REFERENCE = {
    # Cited, frozen, StratifiedGroupKFold(5) pooled-OOF reference of the protocol model
    # (not the federated train/validation/test split used here).
    "auroc": 0.7090, "auroc_ci": [0.6971, 0.7208], "auprc": 0.3830,
    "accuracy": 0.6499, "balanced_accuracy": 0.6471,
    "sensitivity": 0.6421, "specificity": 0.6520,
    "protocol": "StratifiedGroupKFold(5) pooled-OOF over ALL participants -- NOT this session's fixed train/val/test split",
}


def load_arm_b_frame() -> tuple[pd.DataFrame, list[str], pd.Series]:
    """Returns (labeled_frame, feature_cols[152], y_series) -- identical
    construction to scripts/tune_psyche_d_severity_change_models_arm_b.py and
    scripts/final_freeze_psyche_d_severity_change_reproduction.py, re-derived
    here (not imported) since this is a new, standalone module and the
    construction is a 3-line, dependency-free derivation over REPRO's raw
    frame."""
    raw = REPRO.load_raw()
    phase2_base_cols = [c for c in raw.columns if c not in REPRO.LEAKAGE_COLS and c not in ("user_id", "user_mth", "ref_label")]
    assert len(phase2_base_cols) == 150
    feature_cols = phase2_base_cols + BASELINE_SEVERITY_COLS
    assert len(feature_cols) == 152

    labeled = raw[raw["ref_label"]].copy()
    delta_cat = labeled["phq9_cat_end"] - labeled["phq9_cat_start"]
    y = (delta_cat > 0).astype(int)
    y.index = labeled.index
    return labeled, feature_cols, y


def build_fixed_split(labeled: pd.DataFrame, seed: int = SPLIT_SEED) -> dict:
    """ONE-TIME participant-level train/validation/test split (never K-fold
    rotation). Stratified by each participant's baseline-severity mode
    (`phq9_cat_start`) so val/test remain representative of the severity
    distribution. Whole participants only -- never splits one participant's
    rows across sets (longitudinal data: a participant's multiple months
    must never straddle a split boundary)."""
    participant_severity = labeled.groupby("user_id")["phq9_cat_start"].agg(lambda s: s.mode().iloc[0])
    participants = participant_severity.index.to_numpy()
    strata = participant_severity.to_numpy()

    train_p, holdout_p, strata_train, strata_holdout = train_test_split(
        participants, strata, train_size=TRAIN_FRAC, random_state=seed, stratify=strata,
    )
    rel_val = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    val_p, test_p = train_test_split(
        holdout_p, train_size=rel_val, random_state=seed + 1, stratify=strata_holdout,
    )

    train_p, val_p, test_p = set(train_p), set(val_p), set(test_p)
    assert not (train_p & val_p), "train/val participant overlap"
    assert not (train_p & test_p), "train/test participant overlap"
    assert not (val_p & test_p), "val/test participant overlap"
    assert train_p | val_p | test_p == set(participants)

    train_idx = labeled.index[labeled["user_id"].isin(train_p)]
    val_idx = labeled.index[labeled["user_id"].isin(val_p)]
    test_idx = labeled.index[labeled["user_id"].isin(test_p)]
    assert set(train_idx) & set(val_idx) == set()
    assert set(train_idx) & set(test_idx) == set()
    assert set(val_idx) & set(test_idx) == set()

    return {
        "train_participants": train_p, "val_participants": val_p, "test_participants": test_p,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "n_train_participants": len(train_p), "n_val_participants": len(val_p), "n_test_participants": len(test_p),
        "n_train_rows": len(train_idx), "n_val_rows": len(val_idx), "n_test_rows": len(test_idx),
        "seed": seed,
    }


def assign_clients(train_participants: set, k: int, seed: int) -> list[set]:
    """Deterministic whole-participant K-way partition of the (already
    train-only, val/test-excluded) participant pool. Same pattern as this
    project's other FL scripts (seeded permutation + np.array_split), but a
    fresh, independent implementation -- this session's train pool is a fixed
    subset, not a fold's rotating train set, so the function signature and
    semantics differ (no fold_idx)."""
    participants = np.array(sorted(train_participants))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(participants)
    site_arrays = np.array_split(shuffled, k)
    sites = [set(a.tolist()) for a in site_arrays]
    total = sum(len(s) for s in sites)
    assert total == len(participants)
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            assert not (sites[i] & sites[j]), f"client {i}/{j} participant overlap"
    return sites


def federated_sufficient_stats(row_groups: list[pd.DataFrame], feature_cols: list[str]) -> dict:
    """Computes global per-feature mean/std by aggregating (count, sum, sumsq)
    across `row_groups` ("clients") -- federated-safe: only these three
    scalars-per-feature ever need to leave a client, never raw rows. Produces
    EXACTLY the same result whether called with 1 group (centralized) or K
    groups (federated) -- this is asserted in the test/verification step, not
    just claimed. NaNs excluded from each client's local (count, sum, sumsq)
    exactly like `SimpleImputer(strategy="mean")` would (mean of observed
    values only)."""
    n_features = len(feature_cols)
    total_count = np.zeros(n_features)
    total_sum = np.zeros(n_features)
    total_sumsq = np.zeros(n_features)
    for g in row_groups:
        X = g[feature_cols].to_numpy(dtype=np.float64)
        mask = ~np.isnan(X)
        total_count += mask.sum(axis=0)
        total_sum += np.nansum(X, axis=0)
        total_sumsq += np.nansum(X * X, axis=0)
    mean = total_sum / np.maximum(total_count, 1)
    var = total_sumsq / np.maximum(total_count, 1) - mean**2
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)
    std[std < 1e-8] = 1.0  # guard constant columns
    return {"mean": mean, "std": std, "per_feature_count": total_count}


def apply_impute_scale(df: pd.DataFrame, feature_cols: list[str], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Impute NaN with `mean` (already computed train-only / federated-safe),
    then standardize with `std`. Never re-fits on the data being transformed."""
    X = df[feature_cols].to_numpy(dtype=np.float64)
    nan_mask = np.isnan(X)
    X = np.where(nan_mask, mean[None, :], X)
    X = (X - mean[None, :]) / std[None, :]
    return X


def bootstrap_ci_auroc(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aurocs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y_true[idx], y_score[idx]
        if len(np.unique(yb)) < 2:
            continue
        aurocs.append(roc_auc_score(yb, sb))
    if not aurocs:
        return float("nan"), float("nan")
    return float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))


def compute_metrics(name: str, y_true: np.ndarray, y_score: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = 42) -> dict:
    y_pred = (y_score >= 0.5).astype(int)
    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ci_lo, ci_hi = bootstrap_ci_auroc(y_true, y_score, n_boot=n_boot, seed=seed)
    print(f"{name:40s} AUROC={auroc:.4f} [{ci_lo:.4f},{ci_hi:.4f}] AUPRC={auprc:.4f} acc={acc:.4f} bal_acc={bal_acc:.4f} sens={sens:.4f} spec={spec:.4f} n={len(y_true)}")
    return {
        "name": name, "n": int(len(y_true)),
        "auroc": auroc, "auroc_ci": [ci_lo, ci_hi], "auprc": auprc,
        "accuracy": acc, "balanced_accuracy": bal_acc,
        "sensitivity": sens, "specificity": spec,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def verify_disjointness(split: dict, client_sites: list[set]) -> dict:
    """Full pairwise disjointness confirmation across every partition
    boundary this task's methodological rule requires: val/test vs any
    client, and client vs client. Returns a dict written verbatim into every
    step's results JSON so the confirmation travels with every result file,
    not just the split-construction one."""
    all_client_participants = set()
    for s in client_sites:
        assert not (s & all_client_participants), "cross-client participant overlap"
        all_client_participants |= s
    assert all_client_participants == split["train_participants"], "client union must equal the exact train pool, nothing more/less"
    assert not (all_client_participants & split["val_participants"])
    assert not (all_client_participants & split["test_participants"])
    assert not (split["val_participants"] & split["test_participants"])
    return {
        "val_test_disjoint": True,
        "clients_pairwise_disjoint": True,
        "client_union_equals_train_pool_exactly": True,
        "n_clients": len(client_sites),
        "client_sizes_participants": [len(s) for s in client_sites],
    }
