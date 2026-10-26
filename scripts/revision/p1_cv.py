#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. `research_datasets/` is gitignored.
"""JMIR revision Phase 1, cross-validated analyses (tasks 1.1, 1.2, 1.3).

All arms use the identical StratifiedGroupKFold(5, shuffle, random_state=42) folds,
the frozen model family (REPRO.phase2_dart_factory: LightGBM DART, 150 trees, depth 4,
drop 0.1, random_state 42), mean imputation and balanced sample weights, exactly as
scripts/ablation_psyche_d_severity_change_feature_sources.py. CIs: participant
bootstrap, 2000 draws, seed 42, shared by every arm (paired differences).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import audit_psyche_d_baseline_reproduction as REPRO  # noqa: E402
import crossfit_psyche_d_two_stage_pipeline as CROSSFIT  # noqa: E402
import rev_common as RC  # noqa: E402

t0 = time.time()
raw = REPRO.load_raw()
phase1_cols = REPRO.build_phase1_input_cols(raw)
base = [c for c in raw.columns if c not in REPRO.LEAKAGE_COLS and c not in ("user_id", "user_mth", "ref_label")]
fmap = pd.read_csv(RC.OUT / "P0_4_psyche_d_feature_groups.csv").set_index("feature")["group"]
WEAR = [c for c in base if fmap[c] == "wearable"]
CTX = [c for c in base if fmap[c] in ("screener", "lifestyle_medication")]
PHQ = ["phq9_score_start", "phq9_cat_start"]
INTER = ["cat_m1", "cat_m2"] + REPRO.PROBA_COLS_M1 + REPRO.PROBA_COLS_M2
assert (len(WEAR), len(CTX)) == (111, 39)

lab = raw[raw["ref_label"]].copy()
g = lab["user_id"].to_numpy()
cat0 = lab["phq9_cat_start"].astype(int).to_numpy()
y = ((lab["phq9_cat_end"] - lab["phq9_cat_start"]) > 0).astype(int).to_numpy()
n = len(y)
emb_npz = np.load(RC.ROOT / "research_datasets/old/psyche_d/derived/moment_small_embeddings.npz")
assert (emb_npz["row_index"] == lab.index.to_numpy().astype(str)).all()
EMB = [f"moment_{i}" for i in range(512)]
emb = pd.DataFrame(emb_npz["emb"], index=lab.index, columns=EMB)

ARMS = {
    # Table 3 rows + text
    "wearable_context_150": base,
    "a_phq_only": PHQ,
    "f_frozen_phq_context_wearable": base + PHQ,
    "intermediate_only": INTER,
    "phq_plus_intermediate": PHQ + INTER,
    "full_frozen_plus_intermediate": base + PHQ + INTER,
    # 1.2 ablation
    "b_wearable_only": WEAR,
    "c_context_only": CTX,
    "d_phq_wearable": PHQ + WEAR,
    "e_phq_context": PHQ + CTX,
    # MOMENT (PSYCHE-D)
    "moment_only": EMB,
    "moment_plus_frozen": base + PHQ + EMB,
}
EXCL_ARMS = {"f_frozen_excl_20_27": base + PHQ, "a_phq_only_excl_20_27": PHQ}
keep = cat0 != 4

oof = {k: np.full(n, np.nan) for k in list(ARMS) + list(EXCL_ARMS) + ["published_2phase_original", "category_rate_rule"]}
skf = StratifiedGroupKFold(n_splits=REPRO.N_SPLITS, shuffle=True, random_state=REPRO.RANDOM_STATE)
for f, (tr, te) in enumerate(skf.split(np.zeros(n), y, g)):
    tr_idx, te_idx = lab.index[tr], lab.index[te]
    tr_p, te_p = sorted(set(g[tr])), set(g[te])
    assert not set(tr_p) & te_p
    unl = ~raw["ref_label"]
    # published 2-phase design ("original" arm of the crossfit script): phase 1 fitted on
    # outer-train labelled rows, intermediate months generated for every unlabelled row
    imp1, clf1 = CROSSFIT.fit_phase1(raw, phase1_cols, raw["ref_label"] & raw["user_id"].isin(tr_p))
    cat_o = raw["phq9_cat_end"].copy(); pr_o = pd.DataFrame(0.0, index=raw.index, columns=REPRO.PROBA_COLS)
    CROSSFIT.generate_intermediate(raw, phase1_cols, imp1, clf1, unl, cat_o, pr_o)
    # cross-fitted intermediate features for the ablation arms (identical to the ablation script)
    cat_c = raw["phq9_cat_end"].copy(); pr_c = pd.DataFrame(0.0, index=raw.index, columns=REPRO.PROBA_COLS)
    CROSSFIT.generate_intermediate(raw, phase1_cols, imp1, clf1, unl & raw["user_id"].isin(te_p), cat_c, pr_c)
    tr_arr = np.array(tr_p)
    for itr, iho in KFold(n_splits=CROSSFIT.INNER_SPLITS, shuffle=True, random_state=REPRO.RANDOM_STATE * 10_000 + f).split(tr_arr):
        i1, c1 = CROSSFIT.fit_phase1(raw, phase1_cols, raw["ref_label"] & raw["user_id"].isin(set(tr_arr[itr])))
        CROSSFIT.generate_intermediate(raw, phase1_cols, i1, c1, unl & raw["user_id"].isin(set(tr_arr[iho])), cat_c, pr_c)
    idx = tr_idx.append(te_idx)
    gen_c = REPRO.build_phase2_generated_features(raw, idx, cat_c, pr_c)
    gen_o = REPRO.build_phase2_generated_features(raw, idx, cat_o, pr_o)
    X = pd.concat([raw.loc[idx, base + PHQ], gen_c, emb.loc[idx]], axis=1)
    Xo = pd.concat([raw.loc[idx, base], gen_o], axis=1)
    yy = pd.Series(y, index=lab.index).loc[idx]
    trm, tem = idx.isin(tr_idx), idx.isin(te_idx)
    sw = compute_sample_weight("balanced", yy[trm])

    def fit(Xdf, cols, trmask, temask, swt):
        imp = SimpleImputer(strategy="mean", keep_empty_features=True)
        clf = REPRO.phase2_dart_factory()
        clf.fit(imp.fit_transform(Xdf.loc[trmask, cols]), yy[trmask], sample_weight=swt)
        return clf.predict_proba(imp.transform(Xdf.loc[temask, cols]))[:, 1]

    for k, cols in ARMS.items():
        oof[k][te] = fit(X, cols, trm, tem, sw)
    oof["published_2phase_original"][te] = fit(Xo, list(Xo.columns), trm, tem, sw)
    # 1.3d: drop the 20-27 baseline category from training and test
    kidx = pd.Series(keep, index=lab.index).loc[idx].to_numpy()
    trk, tek = trm & kidx, tem & kidx
    swk = compute_sample_weight("balanced", yy[trk])
    te_keep = te[keep[te]]
    for k, cols in EXCL_ARMS.items():
        oof[k][te_keep] = fit(X, cols, trk, tek, swk)
    # 1.3b: trivial rule = training-fold worsening rate of the baseline category
    rate = pd.Series(y[tr]).groupby(cat0[tr]).mean()
    oof["category_rate_rule"][te] = pd.Series(cat0[te]).map(rate).fillna(y[tr].mean()).to_numpy()
    print(f"fold {f + 1}/5 done {time.time() - t0:.0f}s", flush=True)

# leakage comparison (audit A11): logistic regression, 150 inputs, grouped vs random row split
X150 = lab[base].to_numpy(float)
for name, splitter in [("leak_participant_split", StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X150, y, g)),
                       ("leak_record_split", StratifiedKFold(5, shuffle=True, random_state=42).split(X150, y))]:
    oof[name] = np.full(n, np.nan)
    for tr, te in splitter:
        pipe = Pipeline([("imp", SimpleImputer(strategy="mean", keep_empty_features=True)), ("sc", StandardScaler()),
                         ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))])
        pipe.fit(X150[tr], y[tr])
        oof[name][te] = pipe.predict_proba(X150[te])[:, 1]

pd.DataFrame({"user_id": g, "row_id": lab.index, "y": y, "cat_start": cat0, **oof}).to_csv(RC.CACHE / "P1_cv_oof.csv", index=False)

# ---------------- bootstrap ----------------
draws = RC.participant_draws(g)
S = {}
for k, p in oof.items():
    mask = keep if k in EXCL_ARMS else None
    S[k] = RC.summarize(y, p, draws, mask)
S["f_frozen_same_rows_excl"] = RC.summarize(y, oof["f_frozen_phq_context_wearable"], draws, keep)
S["a_phq_same_rows_excl"] = RC.summarize(y, oof["a_phq_only"], draws, keep)
print(f"bootstrap done {time.time() - t0:.0f}s", flush=True)

P = {
    "frozen - phq_only": RC.paired(S["f_frozen_phq_context_wearable"], S["a_phq_only"]),
    "frozen - wearable_context_150": RC.paired(S["f_frozen_phq_context_wearable"], S["wearable_context_150"]),
    "phq_only - wearable_context_150": RC.paired(S["a_phq_only"], S["wearable_context_150"]),
    "full_plus_intermediate - frozen": RC.paired(S["full_frozen_plus_intermediate"], S["f_frozen_phq_context_wearable"]),
    "published_2phase - frozen": RC.paired(S["published_2phase_original"], S["f_frozen_phq_context_wearable"]),
    "d-a (phq+wearable) - phq": RC.paired(S["d_phq_wearable"], S["a_phq_only"]),
    "e-a (phq+context) - phq": RC.paired(S["e_phq_context"], S["a_phq_only"]),
    "f-e frozen - (phq+context): added value of wearables": RC.paired(S["f_frozen_phq_context_wearable"], S["e_phq_context"]),
    "c-b context_only - wearable_only": RC.paired(S["c_context_only"], S["b_wearable_only"]),
    "moment_plus_frozen - frozen": RC.paired(S["moment_plus_frozen"], S["f_frozen_phq_context_wearable"]),
    "excl 20-27: frozen - phq_only": RC.paired(S["f_frozen_excl_20_27"], S["a_phq_only_excl_20_27"]),
    "leak: record - participant split": RC.paired(S["leak_record_split"], S["leak_participant_split"]),
}

# 1.3 severity table
labels = ["0-4", "5-9", "10-14", "15-19", "20-27"]
sev = []
for c in range(5):
    m = cat0 == c
    row = {"baseline_phq9": labels[c], "category_code": c, "n_samples": int(m.sum()), "n_participants": int(len(set(g[m]))),
           "n_worsened": int(y[m].sum()), "worsening_rate": float(y[m].mean())}
    if y[m].sum() >= 50 and (1 - y[m]).sum() > 0:
        s = RC.summarize(y, oof["f_frozen_phq_context_wearable"], draws, m)
        row.update({"frozen_auroc_within": s["auroc"], "frozen_auroc_ci": s["auroc_ci"]})
        s2 = RC.summarize(y, oof["a_phq_only"], draws, m)
        row.update({"phq_only_auroc_within": s2["auroc"], "phq_only_auroc_ci": s2["auroc_ci"]})
    sev.append(row)
pd.DataFrame(sev).to_csv(RC.OUT / "P1_3_baseline_severity.csv", index=False)

arm_desc = {k: len(v) for k, v in {**ARMS, **EXCL_ARMS}.items()}
arm_desc.update({"published_2phase_original": len(base) + len(REPRO.GENERATED_COLS), "category_rate_rule": 1,
                 "leak_participant_split": 150, "leak_record_split": 150})
rows = [{"arm": k, "n_inputs": arm_desc.get(k, ""), **{kk: vv for kk, vv in RC.strip(v).items()}} for k, v in S.items()]
pd.DataFrame(rows).to_csv(RC.OUT / "P1_1_2_cv_arms.csv", index=False)
pd.DataFrame([{"comparison": k, **v} for k, v in P.items()]).to_csv(RC.OUT / "P1_1_2_paired_differences.csv", index=False)
res = {"seeds": {"cv": 42, "model": 42, "bootstrap": RC.BOOT_SEED, "n_boot": RC.N_BOOT, "inner_crossfit": "42*10000+fold"},
       "n_rows": n, "n_participants": int(len(set(g))), "arms": {k: RC.strip(v) for k, v in S.items()}, "paired": P,
       "severity": sev, "runtime_sec": round(time.time() - t0)}
(RC.OUT / "P1_cv_results.json").write_text(json.dumps(res, indent=1))
for k, v in S.items():
    print(f"{k:40s} {RC.fmt(v)}  AUPRC {v['auprc']:.3f}")
for k, v in P.items():
    print(f"{k:55s} {v['delta']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
print(pd.DataFrame(sev).to_string())
