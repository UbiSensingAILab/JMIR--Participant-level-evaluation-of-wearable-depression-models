#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), DiSCover Project (NCT03421223), CC BY-NC 4.0,
# independent of REB25-1587_MOD1. `research_datasets/` is gitignored.
"""Participant-level (cluster) bootstrap CIs and paired differences for the 3 main
PSYCHE-D feature-source arms on the identical folds/model as the frozen ablation:
wearable/context only (150), baseline PHQ only (2), and arm B (152).
The ablation's published CIs resample rows; this script resamples participants
(2000 resamples), which is the level the JMIR manuscript reports.
Aggregate numbers only."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_psyche_d_baseline_reproduction as REPRO  # noqa: E402

OUT = Path(__file__).resolve().parent / "psyche_d_participant_bootstrap_arms_results.json"
N_BOOT = 2000

raw = REPRO.load_raw()
base = [c for c in raw.columns if c not in REPRO.LEAKAGE_COLS and c not in ("user_id", "user_mth", "ref_label")]
arms = {"wearable_context_only": base, "baseline_phq_only": ["phq9_score_start", "phq9_cat_start"],
        "arm_B_frozen": base + ["phq9_score_start", "phq9_cat_start"]}
lab = raw[raw["ref_label"]]
g = lab["user_id"].to_numpy()
y = ((lab["phq9_cat_end"] - lab["phq9_cat_start"]) > 0).astype(int).to_numpy()
oof = {k: np.full(len(y), np.nan) for k in arms}
skf = StratifiedGroupKFold(n_splits=REPRO.N_SPLITS, shuffle=True, random_state=REPRO.RANDOM_STATE)
for tr, te in skf.split(np.zeros(len(y)), y, g):
    sw = compute_sample_weight("balanced", y[tr])
    for k, cols in arms.items():
        X = lab[cols].to_numpy(dtype=float)
        imp = SimpleImputer(strategy="mean", keep_empty_features=True)
        clf = REPRO.phase2_dart_factory()
        clf.fit(imp.fit_transform(X[tr]), y[tr], sample_weight=sw)
        oof[k][te] = clf.predict_proba(imp.transform(X[te]))[:, 1]
users, inv = np.unique(g, return_inverse=True)
idx_by_user = [np.where(inv == i)[0] for i in range(len(users))]
rng = np.random.default_rng(REPRO.RANDOM_STATE)
boots = {k: [] for k in arms}; boots_pr = {k: [] for k in arms}
for _ in range(N_BOOT):
    pick = rng.integers(0, len(users), len(users))
    ix = np.concatenate([idx_by_user[i] for i in pick])
    for k in arms:
        boots[k].append(roc_auc_score(y[ix], oof[k][ix])); boots_pr[k].append(average_precision_score(y[ix], oof[k][ix]))
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
res = {"script": "scripts/psyche_d_participant_bootstrap_arms.py", "n_boot": N_BOOT, "level": "participant (cluster) bootstrap",
       "n_rows": int(len(y)), "n_participants": int(len(users)), "arms": {}, "paired": {}}
for k in arms:
    res["arms"][k] = {"auroc": float(roc_auc_score(y, oof[k])), "auroc_ci": ci(boots[k]),
                      "auprc": float(average_precision_score(y, oof[k])), "auprc_ci": ci(boots_pr[k])}
for a, b in [("arm_B_frozen", "baseline_phq_only"), ("arm_B_frozen", "wearable_context_only"), ("baseline_phq_only", "wearable_context_only")]:
    d = np.array(boots[a]) - np.array(boots[b])
    res["paired"][f"{a} - {b}"] = {"delta": res["arms"][a]["auroc"] - res["arms"][b]["auroc"], "ci": ci(d)}
OUT.write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=1))
