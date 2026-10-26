#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. `research_datasets/` is gitignored.
"""Updated analysis plan, Tier 1, cross-validated tasks R1-R5.

Groups (results/revision/P0_4_psyche_d_feature_groups.csv): P = baseline PHQ-9 (2),
S = screener (29), L = lifestyle and medication changes, end month (10),
W = Fitbit, end month (111). Forecasting inputs (R3): P + S + the W and L features of
the month-1 (SM1 = row m-2) and month-2 (SM2 = row m-1) rows of the same participant;
no end-month (SM3) feature.

Frozen model family for every arm (LightGBM DART, 150 trees, depth 4, drop 0.1, seed 42),
mean imputation, balanced weights, identical StratifiedGroupKFold(5, shuffle, 42) folds.
CIs: participant bootstrap, 2000 draws, seed 42, shared by all arms (paired).
The published 2-phase reproduction and MOMENT predictions are reused from
research_datasets/revision_cache/P1_cv_oof.csv (same folds, same row order; scripts/revision/p1_cv.py).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import audit_psyche_d_baseline_reproduction as REPRO  # noqa: E402
import rev_common as RC  # noqa: E402

t0 = time.time()
raw = REPRO.load_raw()
base = [c for c in raw.columns if c not in REPRO.LEAKAGE_COLS and c not in ("user_id", "user_mth", "ref_label")]
fmap = pd.read_csv(RC.OUT / "P0_4_psyche_d_feature_groups.csv").set_index("feature")["group"]
P = ["phq9_score_start", "phq9_cat_start"]
S = [c for c in base if fmap[c] == "screener"]
L = [c for c in base if fmap[c] == "lifestyle_medication"]
W = [c for c in base if fmap[c] == "wearable"]
assert (len(S), len(L), len(W)) == (29, 10, 111)
MED3 = ["med_start", "med_stop", "med_dose"]
TREAT6 = MED3 + ["nonmed_start", "nonmed_stop", "med_nonmed_dnu"]
assert set(TREAT6) <= set(L)

lab = raw[raw["ref_label"]].copy()
g = lab["user_id"].to_numpy()
y = ((lab["phq9_cat_end"] - lab["phq9_cat_start"]) > 0).astype(int).to_numpy()
cat0 = lab["phq9_cat_start"].astype(int).to_numpy()
n = len(y)

# month-1 / month-2 rows (SM1 = m-2, SM2 = m-1) of the same participant
month = raw.set_index(["user_id", "user_mth"])[W + L]
assert month.index.is_unique
feat = {}
for tag, off in (("sm1", 2), ("sm2", 1)):
    keys = list(zip(lab["user_id"], lab["user_mth"] - off))
    blk = month.reindex(keys)
    blk.index = lab.index
    blk.columns = [f"{c}__{tag}" for c in blk.columns]
    feat[tag] = blk
    feat[tag + "_present"] = np.array([k in month.index for k in keys])
X = pd.concat([lab[base + P], feat["sm1"], feat["sm2"]], axis=1)
SMW = [f"{c}__{t}" for t in ("sm1", "sm2") for c in W]
SML = [f"{c}__{t}" for t in ("sm1", "sm2") for c in L]
has_m1, has_m2 = feat["sm1_present"], feat["sm2_present"]
has_any = has_m1 | has_m2

ARMS = {
    "P": P, "S": S, "L": L, "W": W,
    "P+S": P + S, "P+S+L": P + S + L, "P+S+W": P + S + W, "P+S+L+W (frozen)": base + P,  # same column order as the frozen model
    "frozen minus med_start/stop/dose": [c for c in base + P if c not in MED3],
    "frozen minus 6 treatment-change items": [c for c in base + P if c not in TREAT6],
    "R3 forecast: P+S+SM1/SM2 (W+L)": P + S + SMW + SML,
    "R3 forecast: P+S+SM1/SM2 Fitbit only": P + S + SMW,
}
EXCL = {"P+S+L+W (frozen) excl 20-27": base + P, "P excl 20-27": P}
keep = cat0 != 4

oof = {k: np.full(n, np.nan) for k in list(ARMS) + list(EXCL) + ["category rule"]}
fitted = []
skf = StratifiedGroupKFold(n_splits=REPRO.N_SPLITS, shuffle=True, random_state=REPRO.RANDOM_STATE)
for f, (tr, te) in enumerate(skf.split(np.zeros(n), y, g)):
    assert not set(g[tr]) & set(g[te])

    def fit(cols, trm, tem):
        imp = SimpleImputer(strategy="mean", keep_empty_features=True)
        clf = REPRO.phase2_dart_factory()
        Xtr = imp.fit_transform(X.iloc[trm][cols].to_numpy(float))
        clf.fit(Xtr, y[trm], sample_weight=compute_sample_weight("balanced", y[trm]))
        Xte = imp.transform(X.iloc[tem][cols].to_numpy(float))
        return clf, Xte, clf.predict_proba(Xte)[:, 1]

    for k, cols in ARMS.items():
        clf, Xte, p = fit(cols, tr, te)
        oof[k][te] = p
        if k == "P+S+L+W (frozen)":
            fitted.append((clf, te, Xte))
    trk, tek = tr[keep[tr]], te[keep[te]]
    for k, cols in EXCL.items():
        oof[k][tek] = fit(cols, trk, tek)[2]
    rate = pd.Series(y[tr]).groupby(cat0[tr]).mean()
    oof["category rule"][te] = pd.Series(cat0[te]).map(rate).fillna(y[tr].mean()).to_numpy()
    print(f"fold {f + 1}/5 {time.time() - t0:.0f}s", flush=True)

prev = pd.read_csv(RC.CACHE / "P1_cv_oof.csv")
assert (prev["row_id"].astype(str).to_numpy() == lab.index.astype(str).to_numpy()).all()
assert np.allclose(prev["f_frozen_phq_context_wearable"].to_numpy(), oof["P+S+L+W (frozen)"])
for k_new, k_old in [("published 2-phase (reproduction)", "published_2phase_original"), ("MOMENT alone", "moment_only"),
                     ("MOMENT + frozen", "moment_plus_frozen")]:
    oof[k_new] = prev[k_old].to_numpy()
pd.DataFrame({"user_id": g, "row_id": lab.index, "y": y, "cat_start": cat0, "has_sm1": has_m1, "has_sm2": has_m2,
              **oof}).to_csv(RC.CACHE / "R_tier1_cv_oof.csv", index=False)

draws = RC.participant_draws(g)
Sm = {k: RC.summarize(y, p, draws, keep if k in EXCL else None) for k, p in oof.items()}
sub = {k: RC.summarize(y, oof[k], draws, has_any) for k in ["P+S", "R3 forecast: P+S+SM1/SM2 (W+L)", "R3 forecast: P+S+SM1/SM2 Fitbit only",
                                                          "P+S+L+W (frozen)", "published 2-phase (reproduction)"]}
print(f"bootstrap {time.time() - t0:.0f}s", flush=True)

pairs = [
    ("R2", "(P+S) - P: value of the screener", "P+S", "P"),
    ("R2", "(P+S+W) - (P+S): wearables without concurrent self-report", "P+S+W", "P+S"),
    ("R2", "(P+S+L) - (P+S): lifestyle and medication items", "P+S+L", "P+S"),
    ("R2", "(P+S+L+W) - (P+S+L): wearables beyond everything else", "P+S+L+W (frozen)", "P+S+L"),
    ("R2", "frozen - P: the hybrid gain", "P+S+L+W (frozen)", "P"),
    ("R2", "frozen minus med_start/stop/dose - frozen", "frozen minus med_start/stop/dose", "P+S+L+W (frozen)"),
    ("R2", "frozen minus 6 treatment-change items - frozen", "frozen minus 6 treatment-change items", "P+S+L+W (frozen)"),
    ("R3", "forecast (W+L) - (P+S)", "R3 forecast: P+S+SM1/SM2 (W+L)", "P+S"),
    ("R3", "forecast Fitbit only - (P+S)", "R3 forecast: P+S+SM1/SM2 Fitbit only", "P+S"),
    ("R3", "forecast (W+L) - forecast Fitbit only", "R3 forecast: P+S+SM1/SM2 (W+L)", "R3 forecast: P+S+SM1/SM2 Fitbit only"),
    ("R4", "published 2-phase - frozen", "published 2-phase (reproduction)", "P+S+L+W (frozen)"),
    ("R4", "published 2-phase - R3 forecast (W+L)", "published 2-phase (reproduction)", "R3 forecast: P+S+SM1/SM2 (W+L)"),
    ("R4", "published 2-phase - P+S", "published 2-phase (reproduction)", "P+S"),
    ("R1", "MOMENT + frozen - frozen", "MOMENT + frozen", "P+S+L+W (frozen)"),
    ("R5", "excl 20-27: frozen - P", "P+S+L+W (frozen) excl 20-27", "P excl 20-27"),
]
PR = [{"task": t, "comparison": name, **RC.paired(Sm[a], Sm[b])} for t, name, a, b in pairs]
for name, a, b in [("restricted to samples with month 1-2 data: forecast (W+L) - (P+S)", "R3 forecast: P+S+SM1/SM2 (W+L)", "P+S"),
                   ("restricted: forecast Fitbit only - (P+S)", "R3 forecast: P+S+SM1/SM2 Fitbit only", "P+S"),
                   ("restricted: published 2-phase - R3 forecast (W+L)", "published 2-phase (reproduction)", "R3 forecast: P+S+SM1/SM2 (W+L)")]:
    PR.append({"task": "R3/R4", "comparison": name, **RC.paired(sub[a], sub[b])})

# grouped permutation importance (frozen model, shuffle each group as one block, 10 shuffles)
cols_f = ARMS["P+S+L+W (frozen)"]
gidx = {"P": [cols_f.index(c) for c in P], "S": [cols_f.index(c) for c in S],
        "L": [cols_f.index(c) for c in L], "W": [cols_f.index(c) for c in W]}
base_p = oof["P+S+L+W (frozen)"]
rng = np.random.default_rng(42)


def permuted(idx):
    s = np.empty(n)
    for clf, te, Xte in fitted:
        Xp = Xte.copy()
        Xp[:, idx] = Xp[rng.permutation(len(te))][:, idx]
        s[te] = clf.predict_proba(Xp)[:, 1]
    return s


def perm_importance(idx, n_shuf=10):
    shuf = [permuted(idx) for _ in range(n_shuf)]
    point = float(roc_auc_score(y, base_p) - np.mean([roc_auc_score(y, s) for s in shuf]))
    bb = np.array([roc_auc_score(y[ix], base_p[ix]) - np.mean([roc_auc_score(y[ix], s[ix]) for s in shuf]) for ix in draws])
    return {"auroc_drop": point, "ci": RC.ci(bb)}


GP = {k: perm_importance(v) for k, v in gidx.items()}
print(f"grouped permutation {time.time() - t0:.0f}s", flush=True)
# per-feature permutation importance with CIs: start PHQ-9, the next 10 by point estimate in the
# saved run (scripts/psyche_d_permutation_importance_results.json), and the top wearable input
old = json.loads((RC.ROOT / "scripts/psyche_d_permutation_importance_results.json").read_text())
ranked = [k for k, _ in old["top20_permutation"]]
top_w = next(k for k, _ in sorted(old["permutation_auroc_drop"].items(), key=lambda kv: -kv[1]["mean"]) if k in W)
chosen = list(dict.fromkeys(ranked[:11] + [top_w]))
FP = {c: {**perm_importance([cols_f.index(c)]), "group": fmap.get(c, "baseline_phq9"), "saved_point": old["permutation_auroc_drop"][c]["mean"]}
      for c in chosen}
print(f"feature permutation {time.time() - t0:.0f}s", flush=True)

# R5 severity table (adds P+S within category)
labels = ["0-4", "5-9", "10-14", "15-19", "20-27"]
sev = []
for c in range(5):
    m = cat0 == c
    row = {"baseline_phq9": labels[c], "n_samples": int(m.sum()), "n_participants": int(len(set(g[m]))),
           "n_worsened": int(y[m].sum()), "worsening_rate": float(y[m].mean())}
    if y[m].sum() >= 50:
        for k in ["P+S+L+W (frozen)", "P+S", "P"]:
            s_ = RC.summarize(y, oof[k], draws, m)
            row[f"{k} auroc"] = s_["auroc"]
            row[f"{k} ci"] = s_["auroc_ci"]
    sev.append(row)

# forest plot of R2 paired differences
fp_rows = [r for r in PR if r["task"] == "R2"]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for i, r in enumerate(fp_rows[::-1]):
    ax.errorbar(r["delta"], i, xerr=[[r["delta"] - r["ci"][0]], [r["ci"][1] - r["delta"]]], fmt="o", color="#534AB7", capsize=3)
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_yticks(range(len(fp_rows)))
ax.set_yticklabels([r["comparison"].split(":")[0] for r in fp_rows[::-1]], fontsize=7)
ax.set_xlabel("Paired difference in AUROC (95% CI, participant bootstrap)", fontsize=8)
ax.set_title("PSYCHE-D timing-aware ablation (R2)", fontsize=9)
ax.tick_params(axis="x", labelsize=7)
fig.tight_layout()
fig.savefig(RC.OUT / "R2_forest_paired_differences.png", dpi=300)

meta = {"seeds": {"cv": 42, "model": 42, "bootstrap": RC.BOOT_SEED, "n_boot": RC.N_BOOT, "permutation": 42, "n_shuffles": 10},
        "groups": {"P": P, "S": S, "L": L, "W_count": len(W)}, "medication_items_removed": {"MED3": MED3, "TREAT6": TREAT6},
        "month_rows": {"n_samples": n, "has_sm1": int(has_m1.sum()), "has_sm2": int(has_m2.sum()), "has_both": int((has_m1 & has_m2).sum()),
                       "has_any": int(has_any.sum()), "positives_with_any": int(y[has_any].sum())},
        "n_inputs": {k: len(v) for k, v in {**ARMS, **EXCL}.items()}}
pd.DataFrame([{"arm": k, "n_inputs": meta["n_inputs"].get(k, ""), **RC.strip(v)} for k, v in Sm.items()]).to_csv(RC.OUT / "R1_R2_R3_cv_arms.csv", index=False)
pd.DataFrame([{"arm": k, **RC.strip(v)} for k, v in sub.items()]).to_csv(RC.OUT / "R3_restricted_to_month12_samples.csv", index=False)
pd.DataFrame(PR).to_csv(RC.OUT / "R2_R3_R4_paired_differences.csv", index=False)
pd.DataFrame([{"group": k, **v} for k, v in GP.items()]).to_csv(RC.OUT / "R2_grouped_permutation_importance.csv", index=False)
pd.DataFrame([{"feature": k, **v} for k, v in FP.items()]).to_csv(RC.OUT / "R1_permutation_importance_participant_ci.csv", index=False)
pd.DataFrame(sev).to_csv(RC.OUT / "R5_baseline_severity.csv", index=False)
res = {**meta, "arms": {k: RC.strip(v) for k, v in Sm.items()}, "restricted": {k: RC.strip(v) for k, v in sub.items()},
       "paired": PR, "grouped_permutation": GP, "feature_permutation": FP, "severity": sev, "runtime_sec": round(time.time() - t0)}
(RC.OUT / "R_tier1_cv_results.json").write_text(json.dumps(res, indent=1, default=float))
for k, v in Sm.items():
    print(f"{k:45s} {RC.fmt(v)} AUPRC {v['auprc']:.3f} ({v['auprc_ci'][0]:.3f}-{v['auprc_ci'][1]:.3f}) n={v['n']}")
for k, v in sub.items():
    print(f"[restricted n={v['n']}] {k:40s} {RC.fmt(v)}")
for r in PR:
    print(f"{r['task']:5s} {r['comparison'][:70]:70s} {r['delta']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]")
print(json.dumps(GP, indent=0)); print(json.dumps({k: (round(v['auroc_drop'], 4), [round(x, 4) for x in v['ci']]) for k, v in FP.items()}))
print(pd.DataFrame(sev).to_string()); print(json.dumps(meta["month_rows"]))
