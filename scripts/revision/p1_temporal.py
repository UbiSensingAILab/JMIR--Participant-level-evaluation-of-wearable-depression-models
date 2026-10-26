#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. `research_datasets/` is gitignored.
"""JMIR revision 1.1 add-on: participant-level CI for the PSYCHE-D participant+temporal
split (scripts/temporal_validity_psyche_d_severity_change.py, split 2), whose saved CI
resamples rows. Same split and model; only the bootstrap changes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import audit_psyche_d_baseline_reproduction as REPRO  # noqa: E402
import rev_common as RC  # noqa: E402
import temporal_validity_psyche_d_severity_change as TV  # noqa: E402

raw = REPRO.load_raw()
base = [c for c in raw.columns if c not in REPRO.LEAKAGE_COLS and c not in ("user_id", "user_mth", "ref_label")]
cols = base + TV.BASELINE_SEVERITY_COLS
lab = raw[raw["ref_label"]].copy()
lab["y"] = ((lab["phq9_cat_end"] - lab["phq9_cat_start"]) > 0).astype(int)
early, late = lab["user_mth"].isin(TV.EARLY_QUARTERS), lab["user_mth"].isin(TV.LATE_QUARTERS)
late_only = set(lab.loc[late, "user_id"]) - set(lab.loc[early, "user_id"])
te = lab["user_id"].isin(late_only) & late
tr = ~lab["user_id"].isin(late_only)
imp = SimpleImputer(strategy="mean", keep_empty_features=True)
clf = REPRO.phase2_dart_factory()
clf.fit(imp.fit_transform(lab.loc[tr, cols]), lab.loc[tr, "y"], sample_weight=compute_sample_weight("balanced", lab.loc[tr, "y"]))
p = clf.predict_proba(imp.transform(lab.loc[te, cols]))[:, 1]
y = lab.loc[te, "y"].to_numpy()
g = lab.loc[te, "user_id"].to_numpy()
s = RC.strip(RC.summarize(y, p, RC.participant_draws(g)))
train_late_rows = int((tr & late).sum())
res = {"split": "participant+temporal (test = late quarters of participants seen only late)", "n_test": int(len(y)),
       "n_test_participants": int(len(set(g))), "n_train": int(tr.sum()), "train_rows_from_late_quarters": train_late_rows,
       **s, "seeds": {"model": 42, "bootstrap": RC.BOOT_SEED, "n_boot": RC.N_BOOT}}
(RC.OUT / "P1_1_temporal_participant_ci.json").write_text(json.dumps(res, indent=1))
pd.DataFrame([res]).to_csv(RC.OUT / "P1_1_temporal_participant_ci.csv", index=False)
print(json.dumps(res, indent=1))
