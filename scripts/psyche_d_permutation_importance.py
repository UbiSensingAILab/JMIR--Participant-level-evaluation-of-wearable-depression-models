#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), derived from the DiSCover Project
# (NCT03421223), CC BY-NC 4.0, independent of this project's own ethics
# protocol (REB25-1587_MOD1). `research_datasets/` is gitignored.
"""Variable importance for the frozen PSYCHE-D model (arm B: 150 wearable/context
features + phq9_score_start + phq9_cat_start, LightGBM-DART), matching the
procedure used for the HRV + PHQ-9 and RESILIENT protocol runs:

  * permutation importance: on the identical StratifiedGroupKFold(5, shuffle, 42)
    folds as the ablation, each feature is shuffled inside every outer test fold
    (10 shuffles), and the drop in pooled out-of-fold AUROC is averaged;
  * model-native gain importance from each fold's model, averaged over folds
    (descriptive only).

Aggregate numbers only; no participant IDs or rows are written.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_psyche_d_baseline_reproduction as REPRO  # noqa: E402

OUT = Path(__file__).resolve().parent / "psyche_d_permutation_importance_results.json"
N_SHUFFLES = 10
BASELINE_PHQ_COLS = ["phq9_score_start", "phq9_cat_start"]


def main() -> None:
    t0 = time.time()
    raw = REPRO.load_raw()
    base_cols = [c for c in raw.columns if c not in REPRO.LEAKAGE_COLS and c not in ("user_id", "user_mth", "ref_label")]
    assert len(base_cols) == 150
    cols = base_cols + BASELINE_PHQ_COLS
    lab = raw[raw["ref_label"]]
    groups = lab["user_id"].to_numpy()
    y = ((lab["phq9_cat_end"] - lab["phq9_cat_start"]) > 0).astype(int).to_numpy()
    X = lab[cols].to_numpy(dtype=float)
    skf = StratifiedGroupKFold(n_splits=REPRO.N_SPLITS, shuffle=True, random_state=REPRO.RANDOM_STATE)
    fitted, oof, gain = [], np.full(len(y), np.nan), np.zeros(len(cols))
    for tr, te in skf.split(np.zeros(len(y)), y, groups):
        imp = SimpleImputer(strategy="mean", keep_empty_features=True)
        Xtr, Xte = imp.fit_transform(X[tr]), imp.transform(X[te])
        clf = REPRO.phase2_dart_factory()
        clf.fit(Xtr, y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
        oof[te] = clf.predict_proba(Xte)[:, 1]
        g = clf.booster_.feature_importance(importance_type="gain")
        gain += g / g.sum()
        fitted.append((clf, te, Xte))
    gain /= len(fitted)
    base_auc = roc_auc_score(y, oof)
    print(f"baseline pooled OOF AUROC {base_auc:.4f} (frozen arm B reference 0.7090)")
    rng = np.random.default_rng(REPRO.RANDOM_STATE)
    drops = {}
    for j, c in enumerate(cols):
        d = []
        for _ in range(N_SHUFFLES):
            s = np.empty(len(y))
            for clf, te, Xte in fitted:
                Xp = Xte.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                s[te] = clf.predict_proba(Xp)[:, 1]
            d.append(base_auc - roc_auc_score(y, s))
        drops[c] = {"mean": float(np.mean(d)), "sd": float(np.std(d))}
        if j % 25 == 0:
            print(f"{j + 1}/{len(cols)} features, {time.time() - t0:.0f}s")
    top_perm = sorted(drops.items(), key=lambda kv: -kv[1]["mean"])[:20]
    top_gain = sorted(zip(cols, gain.tolist()), key=lambda z: -z[1])[:20]
    out = {"script": "scripts/psyche_d_permutation_importance.py", "model": "frozen arm B LightGBM-DART (152 features)",
           "folds": "StratifiedGroupKFold(5, shuffle=True, random_state=42), grouped by participant",
           "n_shuffles": N_SHUFFLES, "baseline_pooled_oof_auroc": base_auc,
           "permutation_auroc_drop": drops, "top20_permutation": top_perm,
           "gain_importance_normalized_mean_over_folds": dict(zip(cols, gain.tolist())), "top20_gain": top_gain,
           "runtime_sec": time.time() - t0}
    OUT.write_text(json.dumps(out, indent=2))
    print("top permutation:", [(k, round(v["mean"], 4)) for k, v in top_perm[:10]])
    print("top gain:", [(k, round(v, 3)) for k, v in top_gain[:10]])


if __name__ == "__main__":
    main()
